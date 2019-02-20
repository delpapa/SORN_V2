"""" Counting Task Source (from Lazar et. al 2009)

This script contains a source of the form 'abb...bbc'.
"""
import random

import numpy as np

from common import synapses

class CountingSource(object):
    """Counting source (Lazar et. al 2009).

    This source is composed of randomly alternating sequences of size L+2
    of the form 'abb...bbc' and 'dee...eef'.
    """
    def __init__(self, params):

        self.n = params.L
        self.N_u = int(params.N_u)

        # define alphabet (always size n+2)
        self.alphabet = 'ABCDEF'
        self.N_a = len(self.alphabet)
        # lookup is a dictionary containing the 6 letters
        self.lookup = dict(zip(self.alphabet, range(self.N_a)))

        # build sequences
        word1 = 'A'
        word2 = 'D'
        for _ in range(self.n):
            word1 += 'B'
            word2 += 'E'
        word1 += 'C'
        word2 += 'F'
        self.words = [word1, word2]              # different words
        self.probs = np.ones((2, 2))*0.5          # transition probabilities

        # letter and word counters
        self.word_index = 0                      # index for word
        self.ind = 0                             # index within word
        self.glob_ind = 0                        # global index

        # overlap for input neuron pools
        self.overlap = False

    def generate_connection_e(self, par):
        """Generate the W_eu connection matrix

        Parameters:
            par: Bunch
                Main initial sorn parameters
        """
        # choose random input neuron pools
        W = np.zeros((par.N_e, self.N_a))
        available = set(range(par.N_e))
        for a in range(self.N_a):
            temp = random.sample(available, self.N_u)
            W[temp, a] = 1
            if not self.overlap:
                available -= set(temp)

        # always use a full synaptic matrix
        ans = synapses.FullSynapticMatrix(par, (par.N_e, self.N_a))
        ans.W = W

        return ans

    def char(self):
        """
        Return the current alphabet character
        """
        word = self.words[self.word_index]
        return word[self.ind]

    def sequence_ind(self):
        """
        Return current intra-word index
        """
        return self.ind

    def index(self):
        """
        Return character index
        """
        character = self.char()
        ind = self.lookup[character]
        return ind

    def next_word(self):
        """
        Start a new word, with transition probability from probs
        """
        self.ind = 0
        w = self.word_index
        p = self.probs[w, :]
        self.word_index = np.where(np.random.random() <= np.cumsum(p))[0][0]

    def next(self):
        """
        Return next word character or first character of next word
        """
        self.ind += 1
        self.glob_ind += 1

        string = self.words[self.word_index]
        if self.ind >= len(string):
            self.next_word()

        ans = np.zeros(self.N_a)
        ans[self.index()] = 1
        return ans
