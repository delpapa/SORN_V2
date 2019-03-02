""" Random Sequence Source

This script contains a random sequence source.
"""

import random

import numpy as np

from common import synapses


class TextSource(object):
    """
    """

    def __init__(self, params):
        """
        Initialize text source.

        Arguments:
        params -- bunch of simulation parameters from param.py
        """

        # self.file_path= params.file_path
        self.steps_plastic = params.steps_plastic
        self.steps_readout = params.steps_readout
        self.path_to_text = params.path_to_text

        with open(self.path_to_text, "rt") as fin:
             self.corpus = fin.read().replace('\n', '')

        self.corpus = ''.join(self.corpus[:params.max_corpus_size])
        self.corpus = self.corpus.lower()

        self.alphabet = ''.join(sorted(set(self.corpus)))
        self.A = len(self.alphabet)       # alphabet size
        self.N_e = params.N_e
        self.N_u = params.N_u

        # letter counter
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
        W = np.zeros((self.N_e, self.A))
        available = set(range(self.N_e))
        for a in range(self.A):
            temp = random.sample(available, self.N_u)
            W[temp, a] = 1
            available = set([n for n in available if n not in temp])
            assert len(available) > 0,\
                   'Input alphabet too big for non-overlapping neurons'

        # always use a full synaptic matrix
        ans = synapses.FullSynapticMatrix(params, (self.N_e, self.A))
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

    def index_to_symbol(self, index):
        """
        Convert an alphabet index (randing from 0 to A-1) to the corresponding
        symbol.

        Arguments:
        index -- int, alphabet index (NOT to sequence index ind)

        Returns:
        symbol -- char corresponding to the alphabet index
        """

        symbol = self.alphabet[index]
        return symbol

    def next(self):
        """
        Update current index and return next symbol of the corpus, in the form
        of a one-hot array (from Christoph's implementation). TODO: this is very
        ugly, there must be a better way to do implement this.

        Returns:
        one_hot -- one-hot array of size A containing the next input symbol
        """

        self.ind += 1

        # in case the sequence ends, restart it
        # this does not really affect FDT because the corpus is much bigger
        # than the number of simulation steps.
        if self.ind == len(self.corpus):
            self.ind = 0

        one_hot = np.zeros(self.A)
        one_hot[self.alphabet.find(self.corpus[self.ind])] = 1
        return one_hot
