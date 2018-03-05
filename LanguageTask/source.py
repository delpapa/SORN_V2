"""" Random Sequence Source

This script contains a random sequence source.
"""

import random
import sys
sys.path.insert(0, '../data/FDT')
from create_FDT import Fox_drinks_tea as FDT

import numpy as np

from common import synapses


class TextSource(object):
    """
    Text source

    Read text from a .txt file
    """
    def __init__(self, params):

        self.file_path= params.file_path
        self.steps_plastic = params.steps_plastic
        self.steps_readout = params.steps_readout

        # create new .txt file with the FDT sentences
        with open(self.file_path, 'w') as fout:
            fout.write(FDT(5000).full_string())

        # for simplicity, only lower case
        with open(self.file_path, "rt") as fin:
            self.corpus = fin.read().replace('\n', '')
        # only use lowercase
        self.corpus = self.corpus.lower()

        self.alphabet = ''.join(sorted(set(self.corpus)))
        self.A = len(self.alphabet)
        self.N_u = int(params.N_u)               # input pool size

        # letter and word counters
        self.ind = -1                # index in the corpus

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

    def sequence_ind(self):
        return self.ind

    def index_to_symbol(self, index):
        return self.alphabet[index]

    def next(self):
        """Return next symbol of the corpus"""
        self.ind += 1
        # restart sequence after it is over
        if self.ind == len(self.corpus):
            self.ind = 0

        ans = np.zeros(self.A)
        ans[self.alphabet.find(self.corpus[self.ind])] = 1
        return ans
