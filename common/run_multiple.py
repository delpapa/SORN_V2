import os
import sys
sys.path.insert(0, '')
from importlib import import_module

import numpy as np
import matplotlib.pyplot as plt

from common.sorn import Sorn
from common.stats import Stats
from utils import backup_pickle

# variables and values to run: variables must have the same name as in param.py
# these values always overwrite the values in that file
variables = ['L']
values = [
          np.arange(100, 1001, 20),
         ]
# number of repetitions of each experiment (for statistics)
total_runs = 1

# experiment parameters
display_progress = True                          # display progress bar
experiment_tag = ''                              # to mark experiment

################################################################################
#                              SORN simulation                                 #
################################################################################

# 1. import module and create a dictionary with the variables and values to run
exp_dir = import_module(sys.argv[1])
var_dict = dict(zip(variables, values))

# 2. loop over everyting
for var in var_dict.keys():
    for value in var_dict[var]:
        for run in xrange(total_runs):

            # 2.1 load param and experiment files
            params = exp_dir.param
            experiment_file = exp_dir.experiment

            # 2.2 add experiment specific parameters
            params.get_par()
            setattr(params.par, var, value)
            params.get_aux()
            params.aux.display = display_progress
            params.aux.experiment_tag = experiment_tag

            # 3. initialize sorn, source, and stats objects
            experiment = experiment_file.Experiment(params)
            experiment.start()
            sorn = Sorn(params, experiment.inputsource)
            stats = Stats(experiment.stats_tosave, sorn.params)

            # 4. run one experiment once and calculate performance
            for elem in variables:
                print elem, '=', getattr(params.par, elem),
            print '-- Exp.', run + 1

            experiment.run(sorn, stats)

            # 5. save initial sorn parameters and stats objects
            backup_pickle(experiment, stats)
