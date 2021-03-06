""" Random Sequence Source

This script contains a random sequence source.
"""

import random

import numpy as np

from common import synapses


class FDT_GrammarSource(object):
    """
    Fox-drinks-tea (FDT) grammar input source. This source defines a small
    dictionary of the form 'subject' 'verb' 'object' and randomly draws
    sentences from it.
    """

    def __init__(self, params):
        """
        Initialize FDT source.

        Arguments:
        params -- bunch of simulation parameters from param.py
        """

        # self.file_path= params.file_path
        self.steps_plastic = params.steps_plastic
        self.steps_readout = params.steps_readout
        self.dict = params.dictionary

        if self.dict == 'FDT':

            self.verbs = ['eats ',
                          'drinks ']

            self.subjects = ['man ',
                             'woman ',
                             'girl ',
                             'boy ',
                             'child ',
                             'cat ',
                             'dog ',
                             'fox ']

            self.objects_eat = ['meat.',
                                'bread.',
                                'fish.',
                                'vegetables.']
            self.objects_drink = ['milk.',
                                  'water.',
                                  'juice.',
                                  'tea.']

            self.objects = [self.objects_eat, self.objects_drink]

            # create a huge list with all input sentences
            partial_input_string = []
            for _ in range(self.steps_plastic):
                # create a lot of sentences!
                sub = random.choice(self.subjects)
                ver = random.choice(self.verbs)
                if ver == self.verbs[0]:
                    obj = random.choice(self.objects_eat)
                elif ver == self.verbs[1]:
                    obj = random.choice(self.objects_drink)
                partial_input_string.append(sub+ver+obj)

            # all unique input sentences
            self.all_sentences = np.unique(partial_input_string)

            # remove random sentences
            shuffled_unique_sentences = np.unique(partial_input_string)
            np.random.shuffle(shuffled_unique_sentences)

            # TODO: make sure all words appear at least once
            # TODO: sentences must be removed at random, but this was done in
            # the original work
            self.removed_sentences = []
            if params.n_removed_sentences >= 8:
                self.removed_sentences.extend(['woman drinks milk.',
                                          'fox drinks tea.',
                                          'cat eats vegetables.',
                                          'girl eats meat.',
                                          'child eats fish.',
                                          'boy drinks juice.',
                                          'man drinks water.',
                                          'dog eats bread.'])
            if params.n_removed_sentences >= 16:
                self.removed_sentences.extend(['woman eats meat.',
                                              'fox eats bread.',
                                              'cat drinks tea.',
                                              'girl drinks juice.',
                                              'child drinks water.',
                                              'boy eats fish.',
                                              'man eats vegetables.',
                                              'dog drinks milk.'])
            if params.n_removed_sentences >= 24:
                self.removed_sentences.extend(['woman eats vegetables.',
                                              'fox eats fish.',
                                              'cat drinks juice.',
                                              'girl drinks tea.',
                                              'child drinks milk.',
                                              'boy eats bread.',
                                              'man eats meat.',
                                              'dog drinks water.'])
            if params.n_removed_sentences >= 32:
                self.removed_sentences.extend(['woman drinks water.',
                                              'fox drinks juice.',
                                              'cat eats bread.',
                                              'girl eats fish.',
                                              'child eats vegetables.',
                                              'boy drinks tea.',
                                              'man drinks milk.',
                                              'dog eats meat.'])
            if params.n_removed_sentences >= 40:
                self.removed_sentences.extend(['woman drinks tea.',
                                              'fox drinks milk.',
                                              'cat eats meat.',
                                              'girl eats vegetables.',
                                              'child eats bread.',
                                              'boy drinks water.',
                                              'man drinks juice.',
                                              'dog eats fish.'])
            if params.n_removed_sentences >= 48:
                self.removed_sentences.extend(['woman eats fish.',
                                              'fox eats meat.',
                                              'cat drinks milk.',
                                              'girl drinks water.',
                                              'child drinks juice.',
                                              'boy eats vegetables.',
                                              'man eats bread.',
                                              'dog drinks tea.'])
            if params.n_removed_sentences >= 56:
                self.removed_sentences.extend(['woman drinks juice.',
                                              'fox drinks water.',
                                              'cat eats fish.',
                                              'girl eats bread.',
                                              'child eats meat.',
                                              'boy drinks milk.',
                                              'man drinks tea.',
                                              'dog eats vegetables.'])

            input_string = [x for x in partial_input_string if x not in self.removed_sentences]
            self.used_sentences = np.unique(input_string)

        if self.dict == 'SP':

            self.subjects_singular = ['dog ',
                                      'cat ',
                                      'pig ',
                                      'bird ',
                                      'sun ',
                                      'fish ',
                                      'wind ',
                                      'monkey ',
                                      'horse ',
                                      'lion ',
                                      'elephant ',
                                      'duck ']
            self.subjects_plural = [s[:-1]+'s ' for s in self.subjects_singular]
            self.subjects = self.subjects_singular + self.subjects_plural
            self.subjects = np.array(self.subjects)

            self.verbs_plural =  ['bark.',
                                  'meow.',
                                  'oink.',
                                  'sing.',
                                  'shine.',
                                  'swim.',
                                  'blow.',
                                  'climb.',
                                  'gallop.',
                                  'hunt.',
                                  'toot.',
                                  'quack.']
            self.verbs_singular = [s[:-1]+'s.' for s in self.verbs_plural]
            self.verbs = self.verbs_singular + self.verbs_plural
            self.verbs = np.array(self.verbs)

            # create a huge list with all input sentences
            partial_input_string = []
            for _ in xrange(self.steps_plastic):
                # create a lot of sentences!
                sub = random.choice(self.subjects)
                ver = self.verbs[np.where(self.subjects==sub)][0]
                partial_input_string.append(sub+ver)

            # all unique input sentences
            self.all_sentences = [s+v for s,v in zip(self.subjects_singular, self.verbs_singular)]+\
                                 [s+v for s,v in zip(self.subjects_plural, self.verbs_plural)]
            input_string = [x for x in partial_input_string]
            self.used_sentences = np.unique(input_string)
            self.removed_sentences = []

            # grammatical errors
            all_combinations = [s+v for s in self.subjects_singular for v in self.verbs_singular]+\
                               [s+v for s in self.subjects_plural for v in self.verbs_plural]
            all_wrong = [s+v for s in self.subjects_singular for v in self.verbs_plural]+\
                        [s+v for s in self.subjects_plural for v in self.verbs_singular]
            self.grammatical_errors = [s+v for s,v in zip(self.subjects_singular, self.verbs_plural)]+\
                                      [s+v for s,v in zip(self.subjects_plural, self.verbs_singular)]
            self.grammatical_errors += [s for s in all_wrong if s not in self.grammatical_errors]

            # semantic errors
            self.semantic_errors = [s for s in all_combinations if s not in self.all_sentences]
            sing_errors = [s+v for s,v in zip(self.subjects_singular, self.verbs_plural)]
            plur_errors = [s+v for s,v in zip(self.subjects_plural, self.verbs_singular)]
            self.semantic_errors += [s for s in all_wrong if s not in self.semantic_errors
                                                          if s not in sing_errors
                                                          if s not in plur_errors]

        if self.dict == 'eFDT':

            self.verbs_singular = ['eats ',
                                   'drinks ']
            self.verbs_plural = ['eat ',
                                 'drink ']

            self.subjects_singular = ['man ',
                                      'woman ',
                                      'girl ',
                                      'boy ',
                                      'child ',
                                      'cat ',
                                      'dog ',
                                      'fox ']
            self.subjects_plural = ['men ',
                                    'women ',
                                    'girls ',
                                    'boys ',
                                    'children ',
                                    'cats ',
                                    'dogs ',
                                    'foxes ']

            self.objects_eat = ['meat.',
                                'bread.',
                                'fish.',
                                'vegetables.']
            self.objects_drink = ['milk.',
                                  'water.',
                                  'juice.',
                                  'tea.']

            self.objects = [self.objects_eat, self.objects_drink]

            # create a huge list with all input sentences
            partial_input_string = []
            for _ in xrange(self.steps_plastic):
                # create a lot of sentences!
                temp = random.choice(['sing', 'plur'])
                if temp == 'sing':
                    sub = random.choice(self.subjects_singular)
                    ver = random.choice(self.verbs_singular)
                    if ver == self.verbs_singular[0]:
                        obj = random.choice(self.objects_eat)
                    elif ver == self.verbs_singular[1]:
                        obj = random.choice(self.objects_drink)
                elif temp == 'plur':
                    sub = random.choice(self.subjects_plural)
                    ver = random.choice(self.verbs_plural)
                    if ver == self.verbs_plural[0]:
                        obj = random.choice(self.objects_eat)
                    elif ver == self.verbs_plural[1]:
                        obj = random.choice(self.objects_drink)

                partial_input_string.append(sub+ver+obj)

            # all unique input sentences
            self.all_sentences = np.unique(partial_input_string)
            input_string = [x for x in partial_input_string]
            self.used_sentences = np.unique(input_string)
            self.removed_sentences = []

        self.corpus = ' '.join(input_string)
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
