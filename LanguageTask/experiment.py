""" Random Sequence experiment

This script contains the experimental instructions for the Random Sequence
experiment.
"""

import copy

import numpy as np
from sklearn import linear_model

from source import TextSource

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
        self.inputsource = TextSource(self.init_params)

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
        sorn.params.par.eta_ip = 'off'
        sorn.simulation(stats, phase='train')

        # 3. input without plasticity - test performance (STDP and IP off)
        if display:
            print '\nReadout testing phase:'

        sorn.simulation(stats, phase='test')

        # 4. calculate performance
        if display:
            print '\nCalculating performance using Logistic Regression...',

        # load stats to calculate the performance
        t_train = sorn.params.aux.steps_readouttrain
        t_test = sorn.params.aux.steps_readouttest

        t_past_max = 20
        stats.t_past = np.arange(t_past_max)
        stats.performance = np.zeros(t_past_max)
        for t_past in xrange(t_past_max):
            X_train = stats.raster_readout[t_past:t_train]
            y_train = stats.input_readout[:t_train-t_past].T.astype(int)

            X_test = stats.raster_readout[t_train+t_past:t_train+t_test]
            y_test = stats.input_readout[t_train:t_train+t_test-t_past].T.astype(int)

            readout = linear_model.LogisticRegression()
            output_weights = readout.fit(X_train, y_train)
            stats.performance[t_past] = output_weights.score(X_test, y_test)

        if display:
            print 'done'
