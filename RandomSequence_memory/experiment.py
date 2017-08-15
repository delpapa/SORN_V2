from __future__ import division
import random as randomstr

import numpy as np
import sklearn
from sklearn import linear_model

from common.sources import RandomSequenceSource

class Experiment():

    def __init__(self,params):

            self.params = params
            # keep track of the original parameters
            self.init_eta_stdp = params.W_ee.eta_stdp
            self.init_eta_ip = params.eta_ip

    def start(self):

        alphabet_total = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
        alphabet = alphabet_total[:self.params.A]

        while True:
            sequence = ''.join(randomstr.choice(alphabet) for x in range(self.params.L))
            counter = 0
            for c in alphabet:
                if not c in sequence:
                    counter += 1
            if counter == 0:
                break

        self.inputsource = RandomSequenceSource(sequence, self.params.N_u_e)

        stats_tosave = ['CountingLetterStat',
                        'CountingActivityStat']

        return (self.inputsource, stats_tosave)

    def run(self, sorn, stats):

        # 1. input with plasticity
        if self.params.display:
            print '\nPlasticity phase:'

        sorn.simulation(self.params.steps_plastic, stats, phase='plastic')

        # 2. input without plasticity - train (STDP and IP off)
        if self.params.display:
            print '\nReadout training phase:'

        sorn.W_ee.c.eta_stdp = 'off'
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

        t_past_max = 20
        sorn.params.RS_performance = []
        sorn.params.t_past = []
        for t_past in xrange(t_past_max):
            X_train = stats.activity[t_past:t_train].T
            y_train = stats.letters[:t_train-t_past].T

            X_test = stats.activity[t_train+t_past:t_train+t_test].T
            y_test = stats.letters[t_train:t_train+t_test-t_past].T

            # Logistic Regression: same way that AH did, withletters
            #                      as labels instead of the position
            readout =  linear_model.LogisticRegression()
            output_weights = readout.fit(X_train.T, y_train)
            performance = output_weights.score(X_test.T, y_test)

            sorn.params.t_past.append(t_past)
            sorn.params.RS_performance.append(performance)

        if sorn.params.display:
            print 'done'

        # 5. reset parameters to save
        sorn.W_ee.c.eta_stdp = self.init_eta_stdp
        sorn.params.eta_ip = self.init_eta_ip
