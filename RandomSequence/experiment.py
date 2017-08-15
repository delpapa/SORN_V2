import random as randomstr

import numpy as np
import sklearn
from sklearn import linear_model

from common.sources import RandomSequenceSource

class Experiment(object):

    def __init__(self,params):

            self.params = params
            # keep track of the original learning rates
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

        self.inputsource = RandomSequenceSource(sequence, self.params.N_u)

        stats_tosave = [
                        'CountingLetterStat',
                        'CountingActivityStat'
                        ]

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

        X_train = stats.activity[:t_train-1].T
        y_train = stats.letters[1:t_train].T
        y_train_ind = stats.sequence_ind[1:t_train].T

        X_test = stats.activity[t_train:t_train+t_test-1].T
        y_test = stats.letters[1+t_train:t_train+t_test].T
        y_test_ind = stats.sequence_ind[1+t_train:t_train+t_test].T

        readout = linear_model.LogisticRegression()
        output_weights = readout.fit(X_train.T, y_train_ind)
        performance = output_weights.score(X_test.T, y_test_ind)
        sorn.params.RS_performance = performance

        # # Readout training using PI matrix
        # X_train_pinv = np.linalg.pinv(X_train) # MP pseudo-inverse
        # W_trained = np.dot(y_train, X_train_pinv) # least squares
        #
        # # Network prediction with trained weights
        # y_predicted = np.dot(W_trained, X_test)
        #
        # # Performance by Pseudo-Inverse
        # prediction = np.argmax(y_predicted, axis=0)
        # target = np.argmax(y_test, axis=0)
        # performance2 = (prediction == target).sum()/float(len(target))

        if sorn.params.display:
            print 'done'

        # 5. reset parameters to save
        sorn.W_ee.c.eta_stdp = self.init_eta_stdp
        sorn.params.eta_ip = self.init_eta_ip
