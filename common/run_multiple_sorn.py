from __future__ import division
import os
import sys
sys.path.insert(0, '')
from importlib import import_module

import numpy as np
import matplotlib.pyplot as plt

from common.sorn import Sorn
from common.stats import Stats
from utils import backup_pickle

import RandomSequence as exp_dir                 # experiment directory

# parameters to vary during the multiple simulations
L = np.arange(50, 1001, 50)                   # input sequence sizes
A = np.arange(2, 51, 4)                       # input alphabet size
# N = np.arange(1200, 4101, 200)
total_runs = 1                                   # repeat same experiment
display_progress = False                         # display progress bar
experiment_mark = '_capacity'                    # to mark experiment

################################################################################
#                              SORN simulation                                 #
################################################################################
for l in L:

    for a in A:

        for run in range(total_runs):

            # 1. load param and experiment files
            param_file = exp_dir.param
            experiment_file = exp_dir.experiment

            # 2. add experiment specific parameters
            param_file.c.A = a
            # param_file.c.N_e = n
            # param_file.c.N_i =  int(np.floor(0.2*n))                 # inhibitory neurons
            # param_file.c.N = n + param_file.c.N_i
            param_file.c.L = l
            param_file.c.steps_readouttrain = np.maximum(5000, 3*l)
            param_file.c.N_steps =  (param_file.c.steps_plastic
                                     + param_file.c.steps_readouttrain
                                     + param_file.c.steps_readouttest)
            param_file.c.readout_steps = (param_file.c.steps_readouttrain
                                          + param_file.c.steps_readouttest)
            param_file.c.display = display_progress
            param_file.c.experimentmark = experiment_mark

            # 3. initialize sorn, source, and stats objects
            experiment = experiment_file.Experiment(param_file.c)
            (source, stats_tosave) = experiment.start()
            sorn = Sorn(param_file.c,source)
            stats = Stats(stats_tosave, sorn.params)

            # 4. run one experiment once and calculate performance
            print 'Experiment', run + 1, '- L =',l, '; A =', a,
            experiment.run(sorn, stats)

            # 5. save sorn and stats objects
            backup_pickle(sorn.params, stats, save_stats=False, save_dirs=False)

            # 6. reset objects. TODO: how to do it properly with objects?
            del sorn, stats, experiment, source
