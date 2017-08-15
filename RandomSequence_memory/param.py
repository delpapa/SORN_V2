from __future__ import division
import numpy as np

import utils
c = utils.Bunch()

################################################################################
#                           SORN main parameters                               #
################################################################################
c.N_e = 50                                      # excitatory neurons
c.N_i = int(np.floor(0.2*c.N_e))                 # inhibitory neurons
c.N = c.N_e + c.N_i                              # total number of neurons
c.N_u_e = 10                                     # neurons in each input pool

c.eta_ip = 0.001                                 # IP learning rate
c.h_ip = 0.1                                     # target firing rate

c.input_gain = 1                                 # input gain factor

c.W_ee = utils.Bunch(use_sparse=True,
                     lamb = 10,                  # number of out connections
                     eta_stdp = 0.001)           # STDP learning rate
c.W_ei = utils.Bunch(use_sparse=False)
c.W_ie = utils.Bunch(use_sparse=False)

c.T_e_max = 0.5                                  # max initial threshold for E
c.T_e_min = 0                                    # min initial threshold for E
c.T_i_max = 0.5                                  # max initial threshold for I
c.T_i_min = 0                                    # min initial threshold for I

################################################################################
#                           Experiment parameters                              #
################################################################################
c.experiment_name = 'RandomSequence_memory'

c.L = 50000                                      # sequence size
c.A = 4                                          # alphabet size

c.steps_plastic = 50000                          # sorn training time steps
c.steps_readouttrain = 5000   # readout training time steps
c.steps_readouttest = 5000       # readout testing time steps
c.N_steps =  (c.steps_plastic                    # total number of time steps
              + c.steps_readouttrain
              + c.steps_readouttest)
c.readout_steps = (c.steps_readouttrain          # number of readout steps
                   + c.steps_readouttest)
