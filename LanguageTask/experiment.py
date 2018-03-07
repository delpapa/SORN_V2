""" Random Sequence experiment

This script contains the experimental instructions for the Random Sequence
experiment.
"""

import copy

import numpy as np
from sklearn import linear_model

from source import GrammarSource as experiment_source

class Experiment(object):
    """Experiment class.

    It contains the source, the simulation procedure and back up instructions.
    """
    def __init__(self, params):
        """Start the experiment.

        Initialize relevant variables and stats trackers.

        Parameters:
            params: Bunch
                All sorn inital parameters
        """
        # always keep track of initial sorn parameters
        self.init_params = copy.deepcopy(params.par)

        # results directory name
        self.results_dir = (params.aux.experiment_name
                            + params.aux.experiment_tag
                            + '/N' + str(params.par.N_e))

        # define which stats to store during the simulation
        self.stats_tostore = [
            'InputReadoutStat',
            'RasterReadoutStat',
        ]

        # define which parameters and files to save at the end of the simulation
        #     params: save initial main sorn parameters
        #     stats: save all stats trackers
        #     scripts: backup scripts used during the simulation
        self.files_tosave = [
            'params',
            'stats',
            # 'scripts',
        ]

        # load input source
        self.inputsource = experiment_source(self.init_params)

    def run(self, sorn, stats):
        """
        Run experiment once

        Parameters:
            sorn: Bunch
                The bunch of sorn parameters
            stats: Bunch
                The bunch of stats to save

        """
        display = sorn.params.aux.display

        # 1. input with plasticity
        if display:
            print 'Plasticity phase:'

        sorn.simulation(stats, phase='plastic')

        # 2. input without plasticity - train (STDP and IP off)
        if display:
            print '\nReadout training phase:'

        sorn.params.par.eta_stdp = 'off'
        # sorn.params.par.eta_ip = 'off'
        sorn.simulation(stats, phase='train')
        # train readout layer
        if display:
            print '\nTraining readout layer...'
        t_train = sorn.params.aux.steps_readouttrain
        X_train = stats.raster_readout[:t_train-1]
        y_train = stats.input_readout[1:t_train].T.astype(int)
        n_symbols = sorn.source.A
        lg = linear_model.LogisticRegression()
        readout_layer = lg.fit(X_train, y_train)

        # 3. input without plasticity - test performance (STDP and IP off)
        if display:
            print '\nReadout testing phase:'

        sorn.simulation(stats, phase='test')

        if display:
            print '\nTesting readout layer...'
        t_test = sorn.params.aux.steps_readouttest
        X_test = stats.raster_readout[t_train:t_train+t_test-1]
        y_test = stats.input_readout[1+t_train:t_train+t_test].T.astype(int)

        # print and store the performance for each letter
        spec_perf = {}
        for symbol in np.unique(y_test):
            symbol_pos = np.where(symbol == y_test)
            spec_perf[sorn.source.index_to_symbol(symbol)]=\
                                         readout_layer.score(X_test[symbol_pos],
                                                             y_test[symbol_pos])

        # 4. spont_activity (retro feed input)
        if display:
            print '\nSpontaneous phase:'
        # begin with a random input
        symbol = np.random.choice(n_symbols)
        u = np.zeros(n_symbols)
        u[symbol] = 1

        spont_output = ''
        for _ in xrange(sorn.params.par.steps_spont):
            (x, W_ee) = sorn.step(u)
            symbol = int(readout_layer.predict(x.reshape(1,-1)))
            spont_output += sorn.source.index_to_symbol(symbol)
            u = np.zeros(n_symbols)
            u[symbol] = 1

        # 5. calculate parameters to save
        output_sentences = [s+'. ' for s in spont_output.split('. ')][1:-1]
        stats.n_output_sentences = len(output_sentences)
        stats.n_new = len([s for s in output_sentences \
                           if s in sorn.source.removed_sentences])
        stats.n_wrong = len([s for s in output_sentences \
                               if s not in sorn.source.all_sentences])

        # save some storage space
        if hasattr(stats, 'aux'):
            del stats.aux
        if hasattr(stats, 'par'):
            del stats.par
        # stats.spec_perf = spec_perf

        if display:
            print '\ndone'
