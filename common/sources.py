import numpy as np
import random

import utils
import synapses

class CountingSource(object):
    """
    Counting source (Lazar et. al 2009)
        This source is composed of randomly alternating sequences of size L+2
        of the form 'abb...bbc' and 'dee...eef'.
    """
    def __init__(self, words, probs, N_u_e, overlap = False):

        self.word_index = 0                      # index for word
        self.ind = 0                             # index within word
        self.glob_ind = 0                        # global index
        self.words = words                       # different words
        self.probs = probs                       # transition probabilities
        self.N_u_e = int(N_u_e)                  # active per step

        # alphabet: lookup is a dictionary with letters and indices
        self.alphabet = ''.join(sorted(''.join(set(''.join(self.words)))))
        self.N_a = len(self.alphabet)
        self.lookup = dict(zip(self.alphabet,range(self.N_a)))

        # overlap for input neuron pools
        self.overlap = overlap

    def generate_connection_e(self, N_e):
        """
        Generate the W_eu connection matrix

        Parameters:
            N_e: number of excitatory neurons
        """

        # always overlap if there is not enough neuron pools for the alphabet
        if self.N_u_e * self.N_a > N_e:
            self.overlap = True

        # choose random input neuron pools
        W = np.zeros((N_e,self.N_a))
        available = set(range(N_e))
        for a in range(self.N_a):
            temp = random.sample(available,self.N_u_e)
            W[temp,a] = 1
            if not self.overlap:
                available -= set(temp)

        # always use a full synaptic matrix
        ans = synapses.FullSynapticMatrix((N_e,self.N_a))

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
        p = self.probs[w,:]
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

class RandomSequenceSource():
    """
    Source for the counting task.
    Different of words are presented with individual probabilities.
    """
    def __init__(self, sequence, N_u_e, overlap = True):
        """
        Initializes variables.

        Parameters:
            words: list
                The words to present
            probs: matrix
                The probabilities of transitioning between word i and j
                It is assumed that they are summing to 1
            N_u_e: int
                Number of units to receit input for each letter
        """
        self.sequence_index = 0                  # index for sequences
        self.ind = 0                             # index within sequence
        self.glob_ind = 0
        self.sequence = sequence                 # sequence
        self.N_u_e = int(N_u_e)
        self.overlap = overlap

        self.alphabet = "".join(set(self.sequence))
        self.N_a = len(self.alphabet)
        self.lookup = dict(zip(self.alphabet,range(self.N_a)))

        self.reset()

    def generate_connection_e(self,N_e):

        W = zeros((N_e,self.N_a))
        available = set(range(N_e))
        for a in range(self.N_a):
            temp = random.sample(available,self.N_u_e)
            W[temp,a] = 1

            # remove already used neurons, in case of no overlap
            if not self.overlap:
                available -= set(temp)

        ans = synapses.FullSynapticMatrix((N_e,self.N_a))
        ans.W = W

        return ans

    def char(self):
        return self.sequence[self.ind]

    def sequence_ind(self):
        return self.ind

    def index(self):
        character = self.char()
        ind = self.lookup[character]
        return ind

    def next(self):
        self.ind += 1
        self.glob_ind += 1
        if self.ind >= len(self.sequence):
            self.ind = 0
        ans = zeros(self.N_a)
        ans[self.index()] = 1
        return ans

    def reset(self):
        self.ind = 0
        self.glob_ind = 0

class NoSource():
    """
    No input for the spontaneous conditions

    Parameters:
        N_i: int
            Number of input units
    """
    def __init__(self,N_i=1):
        self.N_i = N_i
    def next(self):
        return np.zeros((self.N_i))

    def global_range(self):
        return 1

    def global_index(self):
        return -1

    def generate_connection_e(self,N_e):
        c = utils.Bunch(use_sparse=False,
                        lamb=np.inf,
                        avoid_self_connections=False)
        tmpsyn = synapses.create_matrix((N_e,self.N_i),c)
        tmpsyn.set_synapses(tmpsyn.get_synapses()*0)
        return tmpsyn

        ans = zeros(self.N_a)
        ans[self.index()] = 1
        return ans
