""" Random Sequence Source

This script contains a random sequence source.
"""

import random

import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("../data/MNIST_data/", one_hot=True)

from common import synapses


class MNISTSource(object):
    """
    Text source from an input .txt file.
    """

    def __init__(self, params):
        """
        Initialize MNIST source.

        Arguments:
        params -- bunch of simulation parameters from param.py
        """

        self.file_path = params.file_path
        self.steps_plastic = params.steps_plastic
        self.steps_readout = params.steps_readout

        self.X_train = mnist.train.images
        self.Y_train = mnist.train.labels
        self.X_test = mnist.test.images
        self.Y_test = mnist.test.labels

        self.m, self.n_x = self.X_train.shape
        self.N_e = params.N_e
        self.N_u = params.N_u

        # letter and word counters
        self.ind = -1                # index in the corpus

    def generate_connection_e(self, params):
        """
        Generate the W_eu connection matrix. TODO: params should not be passed
        as an argument again!

        Arguments:
        params -- bunch of simulation parameters from param.py

        Returns:
        W_eu -- FullSynapticMatrix of size (N_e, A), containing 1s and 0s
        """

        # choose random, non-overlapping input neuron pools
        W = np.zeros((self.N_e, self.n_x))
        available = set(range(self.N_e))
        for a in range(self.n_x):
            temp = random.sample(available, self.N_u)
            W[temp, a] = 1
            available = np.array([n for n in available if n not in temp])
            assert len(available) > 0,\
                   'Input alphabet too big for non-overlapping neurons'

        # always use a full synaptic matrix
        ans = synapses.FullSynapticMatrix(params, (self.N_e, self.n_x))
        ans.W = W

        return ans

    def sequence_ind(self):
        """
        Return the sequence index. The index is the current position in the
        input (here, the whole corpus is considered a sequence). TODO: the names
        'ind' and 'index' are confusing, improve their names.

        Returns:
        ind -- sequence (corpus) index of the current input
        """

        ind = self.ind
        return ind

    def next(self):
        """
        Update current index and return next symbol of the corpus, in the form
        of a one-hot array (from Christoph's implementation). TODO: this is very
        ugly, there must be a better way to do implement this.

        Returns:
        one_hot -- one-hot array of size A containing the next input symbol
        """

        self.ind += 1

        if self.ind == self.n_x:
            self.ind = 0

        one_hot = self.X_train[self.ind, :]
        return one_hot
