import numpy as np
import random

from common import synapses

class NoSource(object):
    """
    Source for the sorn's spontaneous activity
    """
    def __init__(self, params):

        self.N_u = int(params.N_u)

    def generate_connection_e(self, N_e):
        """
        Generate the W_eu connection matrix

        Parameters:
            N_e: number of excitatory neurons
        """
        # choose random array neuron pools
        ans = synapses.FullSynapticMatrix((N_e, 1))
        ans.W = np.zeros(N_e)

        return ans

    def next(self):
        """
        """
        return 0
