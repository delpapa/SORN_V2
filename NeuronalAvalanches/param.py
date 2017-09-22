""" Neuronal Avalanches parameters

This script contains the parameters for the Neuronal Avalanche experiment.
"""

import os

import numpy as np

import utils
par = utils.Bunch()
aux = utils.Bunch()

################################################################################
#                           SORN main parameters                               #
################################################################################
def get_par():
    """ Get main sorn parameters.

    For each sorn simulation, change these parameters manually.
    """
    par.N_e = 200                                  # excitatory neurons
    par.N_u = 0                                    # neurons in each input pool

    par.eta_stdp = 0.004                           # STDP learning rate
    par.prune_stdp = True                          # prune very small weights
    par.eta_istdp = 0.001                          # iSTDP learning rate
    par.eta_ip = 0.01                              # IP learning rate
    par.h_ip = 0.1                                 # target firing rate
    par.sp_prob = 0.1                              # SP probability
    par.sp_init = 0.001                            # SP initial value

    par.input_gain = 1                             # input gain factor

    par.sigma = 0.04                               # noise variance

    par.lamb = 20                                  # number of out connections

    par.T_e_max = 1                                # max initial threshold for E
    par.T_e_min = 0                                # min initial threshold for E
    par.T_i_max = 0.5                              # max initial threshold for I
    par.T_i_min = 0                                # min initial threshold for I

################################################################################
#                           Experiment parameters                              #
################################################################################
    par.steps_plastic = 5000000                    # sorn training time steps

################################################################################
#                    Additional derivative SORN parameters                     #
################################################################################
def get_aux():
    """ Get auxiliary sorn parameters.

    These auxiliary parameters do not have to be changed manually.
    """
    aux.N_i = int(np.floor(0.2*par.N_e))       # inhibitory neurons
    aux.N = par.N_e + aux.N_i             # total number of neurons

    # the experiment_name should be the same name of the directory containing it
    aux.experiment_name = os.path.split(os.path.dirname(\
                                        os.path.realpath(__file__)))[1]

    aux.N_steps = (par.steps_plastic)               # total number of time steps
