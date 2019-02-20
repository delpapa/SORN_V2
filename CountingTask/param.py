""" Counting Task parameters

This script contains the parameters for the Counting Task experiment.
Parameters are divided in two subgroups:
    'par' contains the main model parameters
    'aux' contains auxiliary parameters, which usually do not need to be changed
"""

import os

import numpy as np

import utils
par = utils.Bunch()
aux = utils.Bunch()

# ####################
# SORN main parameters
# ####################
def get_par():
    """ Get main sorn parameters.

    For each sorn simulation, change these parameters manually.
    """
    par.N_e = 200                # excitatory neurons
    par.N_u = 10                 # neurons in each input pool

    par.eta_stdp = 0.001         # STDP learning rate
    par.prune_stdp = True        # prune very small weights
    par.eta_ip = 0.001           # IP learning rate
    par.h_ip = 0.1               # target firing rate

    par.input_gain = 1           # input gain factor

    par.lamb = 10                # number of out connections

    par.T_e_max = 0.5            # max initial threshold for E
    par.T_e_min = 0              # min initial threshold for E
    par.T_i_max = 0.5            # max initial threshold for I
    par.T_i_min = 0              # min initial threshold for I

# #######################
# CountingTask parameters
# #######################
    par.L = 200  # sequence size

    par.steps_plastic = 50000   # sorn training time steps
    par.steps_readout = 5000    # readout train and test steps

# #####################################
# Additional derivative SORN parameters
# #####################################
def get_aux():
    """ Get auxiliary sorn parameters.

    These auxiliary parameters do not have to be changed manually.
    """
    aux.N_i = int(np.floor(0.2*par.N_e))       # inhibitory neurons
    aux.N = par.N_e + aux.N_i             # total number of neurons

    # the experiment_name should be the same name of the directory containing it
    aux.experiment_name = os.path.split(os.path.dirname(\
                                        os.path.realpath(__file__)))[1]
    # training ans testing time steps
    aux.steps_readouttrain = par.steps_readout
    aux.steps_readouttest = par.steps_readout

    aux.N_steps = (par.steps_plastic                # total number of time steps
                   + aux.steps_readouttrain
                   + aux.steps_readouttest)
    aux.readout_steps = (aux.steps_readouttrain     # number of readout steps
                        + aux.steps_readouttest)
