""" The SORN class"""

import sys

import numpy as np
import time

from .synapses import FullSynapticMatrix, SparseSynapticMatrix


class Sorn:
    """
    The famous Self-Organizing Recurrent Neural Network (SORN) class.
    """

    def __init__(self, c, source):
        """
        Initializes sorn variables.

        Arguments:
        c -- Bunch of all sorn parameters from param.py
        source -- The input source
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

        # Initialize the pre-threshold internal state variables
        self.R_x = np.zeros(par.N_e)
        self.R_y = np.zeros(aux.N_i)

        # Initialize thresholds
        self.T_e = par.T_e_min + \
                   np.random.random(par.N_e)*(par.T_e_max-par.T_e_min)
        self.T_i = par.T_i_min + \
                   np.random.random(aux.N_i)*(par.T_i_max-par.T_i_min)

    def step(self, u_new):
        """
        Performs a one-step update of the SORN.

        Arguments:
        u_new -- one-hot array input for the current time step
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
        if par.eta_ip is not 'off':
            self.ip(x_new)
        if par.eta_stdp is not 'off':
            self.W_ee.stdp(self.x, x_new)
            self.W_ee.sn()

        # Apply iSTDP and SP, if necessary
        if hasattr(par, 'eta_istdp') and par.eta_istdp is not 'off':
            self.W_ei.istdp(self.y, x_new)
            self.W_ei.sn()
        if hasattr(par, 'sp_init') and par.sp_init is not 'off':
            self.W_ee.sp()

        # Update SORN variables
        self.x = x_new
        self.y = y_new

    def ip(self, x):
        """
        Apply one step of intrinsic plasticity (IP).

        Arguments:
        x -- current activity array
        """

        if self.params.par.eta_ip is not 'off':
            self.T_e += self.params.par.eta_ip*(x - self.params.par.h_ip)

    def simulation(self, stats, phase='plastic'):
        """
        Sorn simulation for a defined number of steps.

        Arguments:
        stats -- Bunch of stats to store
        phase -- string with the phase of the current simulation
                 possible phases: 'plastic', 'train', 'test'
        """

        source = self.source

        if phase is 'plastic':
            N = self.params.par.steps_plastic
        elif phase is 'train':
            N = self.params.aux.steps_readouttrain
        elif phase is 'test':
            N = self.params.aux.steps_readouttest

        # Simulation loop
        for n in range(N):

            # Simulation step
            u = source.next()
            self.step(u)

            # cache this time step data
            stats.store_step(self.x, u, source, self.W_ee, n, phase)

            # command line progress message
            if self.params.aux.display:
                print('Simulation: {}%\r'.format((int)(n/(N-1.)*100)), end='')
