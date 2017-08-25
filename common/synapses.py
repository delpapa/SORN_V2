import numpy as np
import scipy.sparse as sp

################################################################################
# Synapses: this file contains all the functions to create and update the      #
# weight matrices, including the plasticity rules. STDP can be much more       #
# efficiently implemented using sparse matrices, therefore W_EE is sparse and  #
# the other matrices are dense.                                                #
################################################################################

class FullSynapticMatrix(object):
    """
    Dense connection matrix class for SORN I-E and E-I synapses
    """
    def __init__(self, shape):

        self.W = np.random.rand(*shape)

        # normalize after random initialization
        z = abs(self.W).sum(1)
        self.W /= z[:,None]

    def __mul__(self, x):
        return self.W.dot(x)

class SparseSynapticMatrix(object):
    """
    Sparse connection matrix class for SORN E-E synapses
    Uses the CSC format
    """
    def __init__(self, shape, lamb, eta_stdp):

        self.lamb = lamb
        self.eta_stdp = eta_stdp
        (M,N) = shape

        # probability of connection NOT being present
        p = 1 - lamb/float(N)

        while True:
            # initialize random sparse synapses
            W_ee = np.random.random((M, N))
            W_ee[W_ee <= p] = 0
            W_ee[W_ee > p] = np.random.random((np.sum(W_ee > p)))

            # remove self-connections
            np.fill_diagonal(W_ee, 0)

            # verify if all neurons have at least one incomming synapse
            inc_synaps = np.sum(W_ee, axis = 1)
            if not inc_synaps.__contains__(0):
                break

        # make the matrix sparse
        self.W = sp.csc_matrix(W_ee)

        # normalize after initialization
        z = abs(self.W).sum(1)
        data = self.W.data
        data /= np.array(z[self.W.indices]).reshape(data.shape)

    def stdp(self,from_old,from_new,to_old=None,to_new=None):
        """
        Performs one STDP step (from Christoph's implementation)
        """
        if to_old is None:
            to_old = from_old
        if to_new is None:
            to_new = from_new

        N = self.W.shape[1]
        col = np.repeat(np.arange(N),np.diff(self.W.indptr))
        row = self.W.indices
        data = self.W.data
        data += self.eta_stdp*(to_new[row]*from_old[col] \
                             - to_old[row]*from_new[col])   #Suitable for CSC
        data[data < 0] = 0

    def sn(self):
        """
        Performs synaptic normalization
        """
        z = abs(self.W).sum(1)
        data = self.W.data
        data /= np.array(z[self.W.indices]).reshape(data.shape)

    def __mul__(self,x):
        return self.W * x
