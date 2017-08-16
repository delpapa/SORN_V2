import random as randomstr

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

            # keep track of the original learning rates
            self.init_eta_stdp = params.eta_stdp
            self.init_eta_ip = params.eta_ip

    def start(self, params):
        """
        Start the experiment
        """
        # stats to save at the end of the simulation
        # see common/stats.py for a description of each of them
        stats_tosave = [
                        # 'ActivityStat',
                        'CountingLetterStat',
                        # 'CountingActivityStat',
                        # 'ConnectionFractionStat',
                        'InternalStateStat'
                        ]

        inputsource = CountingSource(params)


        return inputsource, stats_tosave

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
        if sorn.params.display:
            print '\nPlasticity phase:'

        sorn.simulation(sorn.params.steps_plastic, stats, phase='plastic')

        # 2. input without plasticity - train (STDP and IP off)
        if sorn.params.display:
            print '\nReadout training phase:'

        sorn.params.eta_stdp = 'off'
        sorn.params.eta_ip = 'off'
        sorn.simulation(sorn.params.steps_readouttrain, stats, phase='train')

        # 3. input without plasticity - test performance (STDP and IP off)
        if sorn.params.display:
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

        X_test = stats.internal_state[t_train:t_train+t_test].T
        y_test = stats.letters[t_train:t_train+t_test].T

        # Logistic Regression
        readout =  linear_model.LogisticRegression()
        output_weights = readout.fit(X_train.T, y_train)
        performance = output_weights.score(X_test.T, y_test)

        # Performance per letter, if necessary
        performance_letter = np.zeros(sorn.source.N_a)
        for i in xrange(sorn.source.N_a):
            letter = np.where(y_test == i)
            performance_letter[i] = output_weights.score(X_test.T[letter],
                                                         y_test[letter])

        # normalize according to max possible performance (see Lazar et al 2009)
        max_performance = 1 - 1./(2*(sorn.params.L + 2))
        stats.LG_performance = performance/max_performance

        import ipdb; ipdb.set_trace()

        if sorn.params.display:
            print 'done'

        # 5. reset parameters to save
        sorn.params.eta_stdp = self.init_eta_stdp
        sorn.params.eta_ip = self.init_eta_ip
