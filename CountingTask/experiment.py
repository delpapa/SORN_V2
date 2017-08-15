import random as randomstr

import numpy as np
import sklearn
from sklearn import linear_model

from common.sources import CountingSource

class Experiment(object):
    """
    Experiment for the SORN: contains the source, the simulation procedure and
    the performance calculation
    """
    def __init__(self, params):

            self.params = params
            # keep track of the original learning rates
            self.init_eta_stdp = params.eta_stdp
            self.init_eta_ip = params.eta_ip

    def start(self):
        """
        Start the experiment
        """
        # alphabet sequence parameters (always size L+2)
        alphabet = 'ABCDEF'
        word1 = 'A'
        word2 = 'D'
        for i in xrange(self.params.L):
            word1 += 'B'
            word2 += 'E'
        word1 += 'C'
        word2 += 'F'
        probs = np.ones((2,2))*0.5

        # create source
        self.inputsource = CountingSource([word1, word2], probs, \
                                                              self.params.N_u_e)

        # stats to save at the end of the simulation
        # see common/stats.py for a description of each of them
        stats_tosave = [
                        # 'ActivityStat',
                        'CountingLetterStat',
                        # 'CountingActivityStat',
                        # 'ConnectionFractionStat',
                        'InternalStateStat'
                        ]

        return (self.inputsource, stats_tosave)

    def run(self, sorn, stats):
        """
        Run experiment once

        Parameters:
            sorn: Bunch
                The bunch of sorn parameters
            stats: Bunch
                The bunch of stats to save

        """
        # 1. input with plasticity
        if self.params.display:
            print '\nPlasticity phase:'

        sorn.simulation(sorn.params.steps_plastic, stats, phase='plastic')

        # 2. input without plasticity - train (STDP and IP off)
        if self.params.display:
            print '\nReadout training phase:'

        sorn.params.eta_stdp = 'off'
        sorn.params.eta_ip = 'off'
        sorn.simulation(sorn.params.steps_readouttrain, stats, phase='train')

        # 3. input without plasticity - test performance (STDP and IP off)
        if self.params.display:
            print '\nReadout testing phase:'

        sorn.simulation(sorn.params.steps_readouttest, stats, phase='test')

        # 4. calculate performance
        if sorn.params.display:
            print '\nCalculating performance using LG...',

        # load stats to calculate the performance
        t_train = sorn.params.steps_readouttrain
        t_test = sorn.params.steps_readouttest

        X_train = stats.internal_state[:t_train].T
        y_train = stats.letters[:t_train].T
        y_train_ind = stats.sequence_ind[:t_train].T

        X_test = stats.internal_state[t_train:t_train+t_test].T
        y_test = stats.letters[t_train:t_train+t_test].T
        y_test_ind = stats.sequence_ind[t_train:t_train+t_test].T

        # Logistic Regression
        readout =  linear_model.LogisticRegression()
        output_weights = readout.fit(X_train.T, y_train_ind)
        performance = output_weights.score(X_test.T, y_test_ind)

        # normalize according to max possible performance (see Lazar et al 2009)
        max_performance = 1 - 1./(2*(sorn.params.L + 2))
        stats.LG_performance = performance/max_performance

        if self.params.display:
            print 'done'

        # 5. reset parameters to save
        sorn.params.eta_stdp = self.init_eta_stdp
        sorn.params.eta_ip = self.init_eta_ip
