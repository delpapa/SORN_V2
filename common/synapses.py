"""Synapses matrices

This script contains all the functions to create and update the weight matrices,
including the plasticity rules STDP, iSTDP, SN and SP. As a rule, W_EE is sparce
and W_IE, W_IE, W_EU are dense.
"""

import numpy as np
import scipy.sparse as sp

class FullSynapticMatrix(object):
    """Dense connection matrix class for SORN I-E and E-I synapses.

    This class contains every synaptic plasticity related method.
    """
    def __init__(self, par, shape):
        """Creates a random normalized matrix

        Parameters:
            par: Bunch
                Main initial sorn parameters

            aux: Bunch
                Auxiliary initial sorn parameters
        """
        if hasattr(par, 'eta_istdp'):
            self.eta_istdp = par.eta_istdp
        self.h_ip = par.h_ip

        self.W = np.random.random(shape)
        z = abs(self.W).sum(1)
        self.W /= z[:, None] # normalize after random initialization

    def istdp(self, y_old, x):
        """Performs one iSTDP step (from Christoph's implementation)"""
        self.W += -self.eta_istdp*((1-(x[:, None]*(1+1.0/self.h_ip)))\
                                 *y_old[None, :])
        self.W[self.W <= 0] = 0.001
        self.W[self.W > 1.0] = 1.0

    def sn(self):
        """Performs synaptic normalization"""
        z = abs(self.W).sum(1)
        self.W = self.W / z[:, np.newaxis]

    def __mul__(self, x):
        """Shorter matrix-array multiplication"""
        return self.W.dot(x)

class SparseSynapticMatrix(object):
    """Sparse connection matrix class for SORN E-E synapses.

    Uses the CSC format.
    """
    def __init__(self, par):
        """Creates a random, sparse and normalized matrix

        Parameters:
            par: Bunch
                Main initial sorn parameters

            aux: Bunch
                Auxiliary initial sorn parameters
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
            W_ee[W_ee <= p_not] = 0
            W_ee[W_ee > p_not] = np.random.random((np.sum(W_ee > p_not)))

            # remove self-connections
            np.fill_diagonal(W_ee, 0)

            # verify that all neurons have at least one incomming synapse
            inc_synaps = np.sum(W_ee, axis=1)
            if not inc_synaps.__contains__(0):
                break

        # make the matrix sparse
        self.W = sp.csc_matrix(W_ee)
        z = abs(self.W).sum(1) # normalize after initialization
        data = self.W.data
        data /= np.array(z[self.W.indices]).reshape(data.shape)

    def stdp(self, from_old, from_new, to_old=None, to_new=None):
        """Performs one STDP step (from Christoph's implementation)"""
        if to_old is None:
            to_old = from_old
        if to_new is None:
            to_new = from_new

        N = self.W.shape[1]
        col = np.repeat(np.arange(N), np.diff(self.W.indptr))
        row = self.W.indices
        data = self.W.data
        data += self.eta_stdp*(to_new[row]*from_old[col] \
                             - to_old[row]*from_new[col])   #Suitable for CSC
        data[data < 0] = 0

        # prune weights
        if self.prune_stdp:
            self.prune()

    def sn(self):
        """Performs synaptic normalization"""
        z = abs(self.W).sum(1)
        data = self.W.data
        data /= np.array(z[self.W.indices]).reshape(data.shape)

    def sp(self):
        """Performs one SP step"""
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
                print '\nCould not find a new connection\n'

    def prune(self):
        """Prune very small weights"""
        self.W.data[self.W.data < 1e-10] = 0. # eliminate small weights
        self.W.eliminate_zeros()

    def __mul__(self, x):
        """Shorter matrix-array multiplication"""
        return self.W * x
