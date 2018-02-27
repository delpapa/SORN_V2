"""" Random Sequence Source

This script contains a random sequence source.
"""

import random

import numpy as np

from common import synapses

class TextSource(object):
    """
    Random Sequence source
    """
    def __init__(self, params):

        self.file_path = params.file_path
        with open(self.file_path, "rt") as fin:
            self.corpus = fin.read().replace('\n', '')[:params.steps_plastic]

        self.alphabet = ''.join(set(self.corpus))
        self.A = len(self.alphabet)
        # TODO: preprocess chars properly, removing +-= chars
        self.N_u = int(params.N_u)               # input pool size

        # letter and word counters
        self.ind = 0                # index in the corpus

    def generate_connection_e(self, par):
        """
        Generate the W_eu connection matrix

        Parameters:
            N_e: number of excitatory neurons
        """

        # choose random, overlapping input neuron pools
        W = np.zeros((par.N_e, self.A))
        available = set(range(par.N_e))
        # TODO: be sure that the input pools are not equal - random.choice
        for a in range(self.A):
            temp = random.sample(available, self.N_u)
            W[temp, a] = 1

        # always use a full synaptic matrix
        ans = synapses.FullSynapticMatrix(par, (par.N_e, self.A))
        ans.W = W

        return ans

    def next(self):
        """Return next symbol of the corpus"""
        self.ind += 1
        self.corpus[self.ind]

        ans = np.zeros(self.A)
        ans[self.alphabet.find(self.corpus[self.ind])] = 1
        return ans
