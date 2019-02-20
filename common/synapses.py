"""Synapses matrices

This script contains all the functions to create and update the weight
matrices, including the plasticity rules STDP, iSTDP, SN and SP. As a rule,
W_EE is sparce and W_IE, W_IE, W_EU are dense.
"""

import numpy as np
import scipy.sparse as sp
import time


class FullSynapticMatrix:
    """
    Dense connection matrix class for I-E, E-I and E-U synapses.

    This class contains every synaptic plasticity related methods.
    """

    def __init__(self, par, shape):
        """
        Creates a randomly initialized and normalized fully connected matrix

        Arguments:
        par -- Bunch of main sorn parameters
        shape -- Tuple with shape of the matrix to be initialize
        """

        if hasattr(par, 'eta_istdp'):
            self.eta_istdp = par.eta_istdp
        self.h_ip = par.h_ip

        self.W = np.random.random(shape)

        # normalize after random initialization
        if self.W.size > 0:
            z = abs(self.W).sum(1)
            self.W /= z[:,None]

    def istdp(self, y_old, x):
        """
        Apply one iSTDP step (from Christoph's implementation)

        Arguments:
        y_old -- previous inhibitory activity array
        x -- current activity array
        """

        self.W += -self.eta_istdp*((1-(x[:, None]*(1+1.0/self.h_ip)))*
                                   y_old[None, :])

        # remove very small or bigger than 1 weigths, to keep the model stable
        self.W[self.W <= 0] = 0.001
        self.W[self.W > 1.0] = 1.0

    def sn(self):
        """
        Apply one synaptic normalization step
        """

        z = abs(self.W).sum(1)
        self.W /= z[:,None]

    def __mul__(self, x):
        """
        Replace matrix-array multiplication for dot product, in order to make
        the code a big shorter and more readable.
        """

        return self.W.dot(x)


class SparseSynapticMatrix:
    """
    Sparse connection matrix class for SORN E-E synapses.

    Uses the CSC format.
    """

    def __init__(self, par):
        """Creates a random, sparse and normalized matrix

        Arguments:
        par -- Bunch of main initial sorn parameters
        """

        self.lamb = par.lamb
        self.eta_stdp = par.eta_stdp
        self.prune_stdp = par.prune_stdp
        if hasattr(par, 'sp_prob'):
            self.sp_prob = par.sp_prob
        if hasattr(par, 'sp_init'):
            self.sp_init = par.sp_init
        self.N = par.N_e

        # probability of connection NOT being present
        p_not = 1 - self.lamb/float(self.N)

        while True:
            # initialize random sparse synapses
            W_ee = np.random.random((self.N, self.N))
            W_ee[W_ee < p_not] = 0
            W_ee[W_ee > p_not] = np.random.random((np.sum(W_ee > p_not)))

            # remove self-connections
            np.fill_diagonal(W_ee, 0)

            # verify that all neurons have at least one incomming synapse
            inc_synaps = np.sum(W_ee, axis=1)
            if not inc_synaps.__contains__(0):
                break

        # make the matrix sparse
        self.W = sp.csc_matrix(W_ee)
        # normalize after initialization
        z = abs(self.W).sum(1)
        self.W.data /= np.array(z[self.W.indices]).reshape(self.W.data.shape)

    def stdp(self, from_old, from_new, to_old=None, to_new=None):
        """
        Apply one STDP step (from Christoph's implementation)

        Arguments:
        from_old -- activity array from previous time step (pre-synaptic)
        from_new -- activity array from current time step (pre-synaptic)
        to_old -- activity array from previous time step (pos-synaptic)
                  None in case pre and pos synaptic matrix is the same
        to_new -- activity array from current time step (pos-synaptic)
                  None in case pre and pos synaptic matrix is the same
        """

        if to_old is None:
            to_old = from_old
        if to_new is None:
            to_new = from_new

        # Suitable update for CSC
        N = self.W.shape[1]
        col = np.repeat(np.arange(N), np.diff(self.W.indptr))
        row = self.W.indices
        data = self.W.data
        data += self.eta_stdp*(to_new[row]*from_old[col] -
                               to_old[row]*from_new[col])
        data[data < 0] = 0

        # prune weights
        if self.prune_stdp:
            self.prune()

    def sn(self):
        """
        Apply one step of synaptic normalization
        """

        # sklearn normalize does not work here. TODO: why?
        z = abs(self.W).sum(1)
        self.W.data /= np.array(z[self.W.indices]).reshape(self.W.data.shape)

    def sp(self):
        """
        Apply one step of structural plasticity
        """

        if np.random.rand() < self.sp_prob:

            # find new connection
            # try for 1000 times, otherwise ignore SP
            counter = 0
            while True:
                i, j = np.random.randint(self.N, size=2)
                connected = self.W[i, j] > 0
                valid = (i != j)
                if (valid and not connected) or counter == 1000:
                    break
                if valid and connected:
                    counter += 1

            # include new connection
            # temporaly convert to dok for easier update
            if counter < 1000:
                W_dok = self.W.todok()
                W_dok[i, j] = self.sp_init
                self.W = W_dok.tocsc()
            else:
                print('\nCould not find a new connection\n')

    def prune(self):
        """
        Prune very small weights
        """

        self.W.data[self.W.data < 1e-10] = 0.  # eliminate small weights
        self.W.eliminate_zeros()

    def __mul__(self, x):
        """
        Replace matrix-array multiplication for dot product, in order to make
        the code a big shorter and more readable.
        """
        return self.W * x
