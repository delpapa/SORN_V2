import os

import numpy as np

import utils
c = utils.Bunch()

################################################################################
#                           SORN main parameters                               #
################################################################################
c.N_e = 200                                      # excitatory neurons
c.N_i = int(np.floor(0.2*c.N_e))                 # inhibitory neurons
c.N = c.N_e + c.N_i                              # total number of neurons
c.N_u_e = 10                                     # neurons in each input pool

c.eta_stdp = 0.001                               # STDP learning rate
c.eta_ip = 0.001                                 # IP learning rate
c.h_ip = 0.1                                     # target firing rate

c.input_gain = 1                                 # input gain factor

c.lamb = 10                                      # number of out connections

c.T_e_max = 0.5                                  # max initial threshold for E
c.T_e_min = 0                                    # min initial threshold for E
c.T_i_max = 0.5                                  # max initial threshold for I
c.T_i_min = 0                                    # min initial threshold for I

################################################################################
#                           Experiment parameters                              #
################################################################################
# the experiment_name should be the same name of the directory containing it
c.experiment_name = os.path.split(
                        os.path.dirname(
                        os.path.realpath(__file__)))[1]

# experiment parameters for a single run
c.L = 10                                         # sequence size

c.steps_plastic = 50000                          # sorn training time steps
c.steps_readouttrain = np.maximum(5000, 3*c.L)   # readout training time steps
c.steps_readouttest = 5000                       # readout testing time steps
c.N_steps =  (c.steps_plastic                    # total number of time steps
              + c.steps_readouttrain
              + c.steps_readouttest)
c.readout_steps = (c.steps_readouttrain          # number of readout steps
                   + c.steps_readouttest)
