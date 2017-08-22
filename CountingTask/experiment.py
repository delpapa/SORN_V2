import random as randomstr
import copy

import numpy as np
import sklearn
from sklearn import linear_model

from source import CountingSource

class Experiment(object):
    """
    Experiment for the SORN: contains the source, the simulation procedure and
    the performance calculation
    """
    def __init__(self, params):

            # keep track of initial parameters
            self.init_params = copy.deepcopy(params.par)

            # define results dir name
            self.results_dir = (params.aux.experiment_name
                                + params.aux.experiment_tag
                                + '/N' + str(params.par.N_e)
                                + '_L' + str(params.par.L))

    def start(self):
        """
        Start the experiment
        """
        self.stats_tosave = [
                             # 'ActivityReadoutStat',
                             'CountingLetterStat',
                             # 'CountingActivityStat',
                             # 'ConnectionFractionStat',
                             'InternalStateStat'
                            ]

        self.files_tosave = [
                             'params',
                             # 'stats',
                             'performance_only',
                             # 'scripts'
                            ]

        self.inputsource = CountingSource(self.init_params)

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

        X_train = stats.internal_state[:t_train].T
        y_train = (stats.letters[:t_train].T).astype(int)
        y_train_ind = (stats.sequence_ind[:t_train].T).astype(int)

        X_test = stats.internal_state[t_train:t_train+t_test].T
        y_test = (stats.letters[t_train:t_train+t_test].T).astype(int)
        y_test_ind = (stats.sequence_ind[t_train:t_train+t_test].T).astype(int)

        # Logistic Regression
        readout =  linear_model.LogisticRegression()
        output_weights = readout.fit(X_train.T, y_train)
        performance = output_weights.score(X_test.T, y_test)

        # #### Readout training using PI matrix
        #
        # trasform labels in one-hot array
        # onehot_values = np.max(y_train) + 1
        # y_train_onehot = np.eye(onehot_values)[y_train_ind]
        # y_test_onehot = np.eye(onehot_values)[y_test_ind]
        #
        # X_train_pinv = np.linalg.pinv(X_train) # MP pseudo-inverse
        # W_trained = np.dot(y_train_onehot.T, X_train_pinv) # least squares
        #
        # # Network prediction with trained weights
        # y_predicted = np.dot(W_trained, X_test)
        #
        # # Performance by Pseudo-Inverse
        # prediction = np.argmax(y_predicted, axis=0)
        # performance_PI = (prediction == y_test).sum()/float(len(y_test))

        stats.LG_performance = performance

        if display:
            print 'done'
