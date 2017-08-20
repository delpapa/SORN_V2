import random as randomstr
import copy

import numpy as np
import sklearn
from sklearn import linear_model

from source import RandomSequenceSource

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
                                + '_L' + str(params.par.L)
                                + '_A' + str(params.par.A))

            # a initial sanity checks
            assert params.par.L > params.par.A,\
                "Alphabet size A must be smaller than the sequence size L"
            assert params.par.N_e > params.par.N_u,\
                "Input pool size N_u should be smaller than network size N_e"

    def start(self):
        """
        Start the experiment
        """
        self.stats_tosave = [
                             # 'ActivityStat',
                             'CountingLetterStat',
                             'CountingActivityStat'
                             # 'ConnectionFractionStat',
                             # 'InternalStateStat'
                            ]

        self.files_tosave = [
                             # 'params',
                             # 'stats',
                             'performance_only',
                             # 'scripts'
                            ]

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

        sorn.params.par.eta_stdp ='off'
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

        # performance is calculated using the previous time step activity
        X_train = stats.activity[:t_train-1].T
        y_train = stats.letters[1:t_train].T
        y_train_ind = stats.sequence_ind[1:t_train].T

        X_test = stats.activity[t_train:t_train+t_test-1].T
        y_test = stats.letters[1+t_train:t_train+t_test].T
        y_test_ind = stats.sequence_ind[1+t_train:t_train+t_test].T

        readout = linear_model.LogisticRegression()
        output_weights = readout.fit(X_train.T, y_train_ind)
        performance = output_weights.score(X_test.T, y_test_ind)
        stats.LG_performance = performance

        if display:
            print 'done'
