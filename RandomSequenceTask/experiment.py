import random as randomstr

import numpy as np
import sklearn
from sklearn import linear_model

from source import RandomSequenceSource

class Experiment(object):

    def __init__(self,params):

            # keep track of the original learning rates
            self.init_eta_stdp = params.eta_stdp
            self.init_eta_ip = params.eta_ip

    def start(self, params):

        stats_tosave = [
                        # 'ActivityStat',
                        'CountingLetterStat',
                        'CountingActivityStat'
                        # 'ConnectionFractionStat',
                        # 'InternalStateStat'
                        ]

        inputsource = RandomSequenceSource(params)

        return inputsource, stats_tosave

    def run(self, sorn, stats):

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

        import ipdb; ipdb.set_trace()

        if sorn.params.display:
            print 'done'

        # 5. reset parameters to save
        sorn.params.eta_stdp = self.init_eta_stdp
        sorn.params.eta_ip = self.init_eta_ip
