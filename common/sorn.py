import sys

import numpy as np
import scipy.sparse as sp

from synapses import FullSynapticMatrix, SparseSynapticMatrix

class Sorn(object):
    """
    The famous Self-Organizing Recurrent Neural Network (SORN)
    """
    def __init__(self, c, source):
        """
        Initializes the SORN variables

        Parameters:
            c: bunch
                The bunch of sorn parameters from param.py
            source: Source
                The input source
        """
        self.params = c
        self.source = source

        p = self.params.par
        a = self.params.aux

        # Initialize weight matrices
        # W_to_from (W_ie = from excitatory to inhibitory)
        self.W_ee = SparseSynapticMatrix((p.N_e,p.N_e), p.lamb, p.eta_stdp)
        self.W_ie = FullSynapticMatrix((a.N_i,p.N_e))
        self.W_ei = FullSynapticMatrix((p.N_e,a.N_i))
        self.W_eu = self.source.generate_connection_e(p.N_e)

        # Initialize the activation of neurons randomly
        self.x = (np.random.random(p.N_e)<0.5) + 0
        self.y = (np.random.random(a.N_i)<0.5) + 0
        self.u = source.next()

        # Initialize the pre-threshold variables
        self.R_x = np.zeros(p.N_e)
        self.R_y = np.zeros(a.N_i)

        # Initialize thresholds
        self.T_i = p.T_i_min + np.random.random(a.N_i)*(p.T_i_max-p.T_i_min)
        self.T_e = p.T_e_min + np.random.random(p.N_e)*(p.T_e_max-p.T_e_min)

    def step(self, u_new):
        """
        Performs a one-step update of the SORN

        Parameters:
            u_new: array
                The input for this step. 1 for the current input, 0 otherwise
        """
        p = self.params.par

        # Compute new state
        self.R_x = self.W_ee*self.x - self.W_ei*self.y - self.T_e
        if hasattr(p, 'sigma'):
            self.R_x += p.sigma * np.random.rand(p.N_e)

        x_temp = self.R_x + p.input_gain*(self.W_eu*u_new)
        self.x_int = (self.R_x >= 0.0)+0
        x_new = (x_temp >= 0.0)+0

        self.R_y = self.W_ie*x_new - self.T_i
        if hasattr(p, 'sigma'):
            self.R_x += p.sigma * np.random.rand(p.N_e)
        y_new = (self.R_y >= 0.0)+0

        # Apply IP, STDP, SN
        if p.eta_ip != 'off':
            self.ip(x_new)
        if p.eta_stdp != 'off':
            self.W_ee.stdp(self.x, x_new)
            self.W_ee.sn()

        # Apply iSTDP and SP, if necessary
        if hasattr(p, 'eta_istdp') and p.eta_istdp != 'off':

            pass

        if hasattr(p, 'sp_init') and p.sp_init != 'off':

            pass

        self.x = x_new
        self.y = y_new
        self.u = u_new

        # Update statistics
        return self.x, self.x_int, self.W_ee

    def ip(self, x):
        """
        Performs intrinsic plasticity

        Parameters:
            x: array
                The current activity array
        """
        if not self.params.par.eta_ip == 'off':
            self.T_e += self.params.par.eta_ip*(x - self.params.par.h_ip)

    def simulation(self, stats, phase='plastic'):
        """
        Simulates SORN for a defined number of steps

        Parameters:
            stats: bunch
                Bunch of stats to save

            phase: string
                Phase the current simulation is in
                Possible phases: 'plastic', 'train', or 'test'
        """
        source = self.source

        if phase == 'plastic':
            N = self.params.par.steps_plastic

        elif phase == 'train':
            N = self.params.aux.steps_readouttrain

        else:
            N = self.params.aux.steps_readouttest

        # Simulation loop
        for n in xrange(N):

            # Simulation step
            u = source.next()
            (x, x_int, W_ee) = self.step(u)

            # update stats. TODO: this should be done by a stats method instead
            #               TODO: some stats have to be saved for the whole sim
            if phase in ['train', 'test']:
                if phase == 'train':
                    step = n
                if phase == 'test':
                    step = n + self.params.aux.steps_readouttrain

                if hasattr(stats, 'total_activity'):
                    stats.total_activity[step] = x.sum()
                if hasattr(stats, 'connec_frac'):
                    stats.connec_frac[step] = W_ee.W.sum()
                if hasattr(stats, 'activity'):
                    stats.activity[step] = x
                if hasattr(stats, 'letters'):
                    stats.sequence_ind[step] = int(source.sequence_ind())
                    stats.letters[step] = int(np.argmax(u))
                if hasattr(stats, 'internal_state'):
                    stats.internal_state[step] = x_int

            # Command line progress message
            if self.params.aux.display:
                if (N>100) and ((n%((N-1)//100) == 0) or (n == N-1)):
                    sys.stdout.write('\rSimulation: %3d%%'%((int)(n/(N-1.)*100)))
                sys.stdout.flush()
