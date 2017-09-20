"""" No souce: source for spontaneous activity

This script contains an empty source. All inputs are '0'.
"""

import numpy as np

from common import synapses

class NoSource(object):
    """Null source for the sorn's spontaneous activity"""
    def __init__(self, par):

        self.N_u = int(par.N_u)

    def generate_connection_e(self, par):
        """Generate the W_eu connection matrix

        Parameters:
            par: Bunch
                Main initial sorn parameters
        """
        # use a full synaptic matrix
        ans = synapses.FullSynapticMatrix(par, (par.N_e, self.N_u))
        ans.W = np.zeros(par.N_e)

        return ans

    def next(self):
        """NoSource always return 0 here"""
        return 0
