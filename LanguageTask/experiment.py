""" LanguageTask experiment

This script contains the experimental instructions for the LanguageTask
experiment.
"""

import copy
from collections import Counter

import numpy as np
from sklearn import linear_model

from source import FDT_GrammarSource as experiment_source


class Experiment(object):
    """Experiment class.

    It contains the source, the simulation procedure and back-up instructions.
    """
    def __init__(self, params):
        """Start the experiment. Initialize relevant variables and stats
        trackers.

        Arguments:
        params -- Bunch of all sorn inital parameters
        """

        # always keep track of initial sorn parameters
        self.init_params = copy.deepcopy(params.par)
        np.random.seed(42)

        # results directory name
        # folder 'LanguageTask_XX/NXX/'
        self.results_dir = (params.aux.experiment_name
                            + params.aux.experiment_tag
                            + '/N' + str(params.par.N_e))

        # define which stats to cache during the simulation
        self.stats_cache = [
            # 'ActivityStat', # number of active neurons
            # 'ActivityReadoutStat', # the total activity only for the readout
            'ConnectionFractionStat', # the fraction of active E-E connections
            'InputReadoutStat', # the input and input index for the readout
            'RasterReadoutStat', # the raster for the readout
        ]

        # define which parameters and files to save at the end of the simulation
        # params: save initial main sorn parameters
        # stats: save all stats trackers
        # scripts: back-up scripts used during the simulation
        self.files_tosave = [
            'params',
            'stats',
            # 'scripts',
        ]

        # create and load input source
        self.inputsource = experiment_source(self.init_params)

    def run(self, sorn, stats):
        """
        Run experiment once and store parameters and variables to save.

        Arguments:
        sorn -- Bunch of all sorn parameters
        stats -- Bunch of stats to save at the end of the simulation
        """

        display = sorn.params.aux.display

        # Step 1. Input with plasticity
        if display:
            print 'Plasticity phase:'

        sorn.simulation(stats, phase='plastic')

        # Step 2. Input without plasticity: train (with STDP and IP off)
        if display:
            print '\nReadout training phase:'

        sorn.params.par.eta_stdp = 'off'
        # sorn.params.par.eta_ip = 'off'
        sorn.simulation(stats, phase='train')

        # Step 3. Train readout layer with logistic regression
        if display:
            print '\nTraining readout layer...'
        t_train = sorn.params.aux.steps_readouttrain
        X_train = stats.raster_readout[:t_train-1]
        y_train = stats.input_readout[1:t_train].T.astype(int)
        n_symbols = sorn.source.A
        lg = linear_model.LogisticRegression()
        readout_layer = lg.fit(X_train, y_train)
        readout_layer = lg.fit(X_train, y_train)

        # Step 4. Input without plasticity: test (with STDP and IP off)
        if display:
            print '\nReadout testing phase:'

        sorn.simulation(stats, phase='test')

        # Step 5. Estimate SORN performance
        if display:
            print '\nTesting readout layer...'
        t_test = sorn.params.aux.steps_readouttest
        X_test = stats.raster_readout[t_train:t_train+t_test-1]
        y_test = stats.input_readout[1+t_train:t_train+t_test].T.astype(int)

        # store the performance for each letter in a dictionary
        spec_perf = {}
        for symbol in np.unique(y_test):
            symbol_pos = np.where(symbol == y_test)
            spec_perf[sorn.source.index_to_symbol(symbol)]=\
                                         readout_layer.score(X_test[symbol_pos],
                                                             y_test[symbol_pos])

        # Step 6. Generative SORN with spont_activity (retro feed input)
        if display:
            print '\nSpontaneous phase:'

        # begin with the prediction from the last step
        symbol = readout_layer.predict(X_test[-1].reshape(1,-1))
        u = np.zeros(n_symbols)
        u[symbol] = 1

        # update sorn and predict next input
        spont_output = ''
        for _ in xrange(sorn.params.par.steps_spont):
            sorn.step(u)
            symbol = int(readout_layer.predict(sorn.x.reshape(1,-1)))
            spont_output += sorn.source.index_to_symbol(symbol)
            u = np.zeros(n_symbols)
            u[symbol] = 1

        # Step 7. Calculate parameters to save (exclude first and last sentences
        # and separate sentences by '.'. Also, remove extra spaces.
        output_sentences = [s[1:]+'.' for s in spont_output.split('.')][1:-1]

        # all output sentences
        output_dict = Counter(output_sentences)
        stats.n_output = len(output_sentences)

        # new output sentences
        new_dict = Counter([s for s in output_sentences \
                           if s in sorn.source.removed_sentences])
        stats.n_new = sum(new_dict.values())

        # wrong output sentences
        wrong_dict = Counter([s for s in output_sentences \
                               if s not in sorn.source.all_sentences])
        stats.n_wrong = sum(wrong_dict.values())

        # save some storage space by deleting some parameters.
        if hasattr(stats, 'aux'):
            del stats.aux
        if hasattr(stats, 'par'):
            del stats.par
        # stats.spec_perf = spec_perf

        if display:
            print '\ndone!'
