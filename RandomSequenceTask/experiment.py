""" Random Sequence experiment

This script contains the experimental instructions for the Random Sequence
experiment.
"""

import copy

import numpy as np
from sklearn import linear_model

from source import RandomSequenceSource

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
                            + '/N' + str(params.par.N_e)
                            + '_L' + str(params.par.L)
                            + '_A' + str(params.par.A)
                            + '_T' + str(params.par.steps_plastic))

        # define which stats to store during the simulation
        self.stats_cache = [
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
        self.inputsource = RandomSequenceSource(self.init_params)

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

        # if sorn.params.aux.experiment_tag == '_LearningCapacity':
        #     # performance is calculated using the previous time step activity
        #     X_train = stats.raster_readout[:t_train-1].T
        #     y_train_ind = stats.input_index_readout[1:t_train].T
        #
        #     X_test = stats.raster_readout[t_train:t_train+t_test-1].T
        #     y_test_ind = stats.input_index_readout[1+t_train:t_train+t_test].T
        #
        #     readout = linear_model.LogisticRegression()
        #     output_weights = readout.fit(X_train.T, y_train_ind)
        #     performance = output_weights.score(X_test.T, y_test_ind)
        #     stats.performance = performance

        # if sorn.params.aux.experiment_tag == '_FadingMemory':

        t_past_max = 20
        t_code = 10
        stats.t_past = np.arange(t_past_max)
        stats.performance = np.zeros(t_past_max)
        for t_past in xrange(t_code, t_past_max):

            X_train = stats.raster_readout[t_past:t_train]
            y_train = stats.input_readout[:t_train-t_past].T.astype(int)

            X_train_new = []
            for j in range(t_code, X_train.shape[0]):
                X_train_new.append(X_train[j-t_code:j].reshape(X_train.shape[1]*t_code))
            X_train_new = np.array(X_train_new)
            y_train_new = y_train[:-t_code]

            X_test = stats.raster_readout[t_train+t_past:t_train+t_test]
            y_test = stats.input_readout[t_train:t_train+t_test-t_past].T.astype(int)

            X_test_new = []
            for j in range(t_code, X_test.shape[0]):
                X_test_new.append(X_test[j-t_code:j].reshape(X_test.shape[1]*t_code))
            X_test_new = np.array(X_test_new)
            y_test_new = y_test[:-t_code]

            readout = linear_model.LogisticRegression()
            output_weights = readout.fit(X_train_new, y_train_new)
            stats.performance[t_past] = output_weights.score(X_test_new, y_test_new)

        if display:
            print 'done'
