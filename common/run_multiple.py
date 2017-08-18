import os
import sys
sys.path.insert(0, '')
from importlib import import_module

import numpy as np
import matplotlib.pyplot as plt

from common.sorn import Sorn
from common.stats import Stats
from utils import backup_pickle

import RandomSequenceTask as exp_dir             # experiment directory

# variables and values to run: variables must have the same name as in param.py
# these values always overwrite the values in that file
variables = ['L', 'A']
values = [
          np.array([100]),
          np.array([4])
         ]
# number of repetitions of each experiment (for statistics)
total_runs = 1

# experiment parameters
display_progress = True                          # display progress bar
experiment_tag = ''                              # to mark experiment

################################################################################
#                              SORN simulation                                 #
################################################################################

# 1. create a dictionary with the variables and values to run and
var_dict = dict(zip(variables, values))

# 2. loop over everyting
for var in var_dict.keys():
    for value in var_dict[var]:
        for run in xrange(total_runs):

            # 2.1 load param and experiment files
            params = exp_dir.param
            experiment_file = exp_dir.experiment

            # 2.2 add experiment specific parameters
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
            import ipdb; ipdb.set_trace()
            print 'Experiment', run + 1, '--',
            for elem in variables:
                print elem, '=', getattr(params.par, elem),
            print '\n'

            experiment.run(sorn, stats)

            # 5. save initial sorn parameters and stats objects
            backup_pickle(experiment.init_params,
                          experiment.results_dir,
                          stats,
                          save_stats=False,
                          save_dirs=False)

            # 6. reset objects. TODO: how to do it properly with objects?
            del sorn, stats, experiment, experiment_file, params
