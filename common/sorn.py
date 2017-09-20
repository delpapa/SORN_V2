import sys

import numpy as np

from synapses import FullSynapticMatrix, SparseSynapticMatrix


class Sorn(object):
    """The famous Self-Organizing Recurrent Neural Network (SORN) class"""
    def __init__(self, c, source):
        """Initializes sorn variables

        Parameters:
            c: bunch
                The bunch of sorn parameters from param.py
            source: Source
                The input source
        """
        self.params = c
        self.source = source

        par = self.params.par
        aux = self.params.aux

        # Initialize weight matrices
        # W_to_from (W_ie = from excitatory to inhibitory)
        self.W_ee = SparseSynapticMatrix(par)
        self.W_ie = FullSynapticMatrix(par, (aux.N_i, par.N_e))
        self.W_ei = FullSynapticMatrix(par, (par.N_e, aux.N_i))
        self.W_eu = self.source.generate_connection_e(par)

        # Initialize the activation of neurons randomly
        self.x = (np.random.random(par.N_e) < 0.5) + 0
        self.y = (np.random.random(aux.N_i) < 0.5) + 0
        self.u = source.next()

        # Initialize the pre-threshold variables
        self.R_x = np.zeros(par.N_e)
        self.R_y = np.zeros(aux.N_i)

        # Initialize thresholds
        self.T_e = par.T_e_min + \
                   np.random.random(par.N_e)*(par.T_e_max-par.T_e_min)
        self.T_i = par.T_i_min + \
                   np.random.random(aux.N_i)*(par.T_i_max-par.T_i_min)

    def step(self, u_new):
        """
        Performs a one-step update of the SORN

        Parameters:
            u_new: array
                The input for this step. 1 for the current input, 0 otherwise
        """
        par = self.params.par
        aux = self.params.aux

        # Compute new state
        self.R_x = self.W_ee*self.x - self.W_ei*self.y - self.T_e
        if hasattr(par, 'sigma'):
            self.R_x += par.sigma*np.random.rand(par.N_e)
        x_temp = self.R_x + par.input_gain*(self.W_eu*u_new)
        self.x_int = (self.R_x >= 0.0)+0
        x_new = (x_temp >= 0.0)+0

        self.R_y = self.W_ie*x_new - self.T_i
        if hasattr(par, 'sigma'):
            self.R_y += par.sigma*np.random.rand(aux.N_i)
        y_new = (self.R_y >= 0.0)+0

        # Apply IP, STDP, SN
        if par.eta_ip != 'off':
            self.ip(x_new)
        if par.eta_stdp != 'off':
            self.W_ee.stdp(self.x, x_new)
            self.W_ee.sn()

        # Apply iSTDP and SP, if necessary
        if hasattr(par, 'eta_istdp') and par.eta_istdp != 'off':
            self.W_ei.istdp(self.y, x_new)
            # self.W_ei.sn()
        if hasattr(par, 'sp_init') and par.sp_init != 'off':
            self.W_ee.sp()

        self.x = x_new
        self.y = y_new
        self.u = u_new

        # Update statistics
        return self.x, self.W_ee

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
        """Sorn simulation for a defined number of steps.

        Parameters:
            stats: Bunch
                Bunch of stats to store

            phase: string
                Phase of the current simulation
                Possible phases: 'plastic', 'train', or 'test'
        """
        source = self.source

        if phase == 'plastic':
            N = self.params.par.steps_plastic
        elif phase == 'train':
            N = self.params.aux.steps_readouttrain
        elif phase == 'test':
            N = self.params.aux.steps_readouttest

        # Simulation loop
        for n in xrange(N):

            # Simulation step
            u = source.next()
            (x, W_ee) = self.step(u)

            # store step data
            stats.store_step(x, u, source, W_ee, n, phase)

            # command line progress message
            if self.params.aux.display:
                if (N > 100) and ((n % ((N-1)//100) == 0) or (n == N-1)):
                    sys.stdout.write('\rSimulation: %3d%%'
                                     % ((int)(n/(N-1.)*100)))
                sys.stdout.flush()
