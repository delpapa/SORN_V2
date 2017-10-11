"""This script runs multiple sorn simulations for the experiment given as an
argument with the specified parameters and experiment instructions. parameters
can be varied here, using 'variables' and 'values'"""

import sys
sys.path.insert(0, '')  # this is needed for the cluster. TODO: why?
from importlib import import_module

import numpy as np

from common.sorn import Sorn
from common.stats import Stats
from utils import backup_pickle

# variables and values to run: variables must have the same name as in param.py
# these values always overwrite the values in that file
VARIABLES = [
    'sigma',
]
VALUES = [
    np.array([0, 0.005, 0.05, 0.5, 5]),
]

# number of repetitions of each experiment (for statistics)
RUNS = 5

# experiment parameters
DISPLAY_PROGRESS = False                         # display progress bar
EXPERIMENT_TAG = '_PZ_hugegain'                              # to mark experiment

################################################################################
#                              SORN simulation                                 #
################################################################################

# 1. import module and create a dictionary with the variables and values to run
assert len(sys.argv) > 1, \
    "Experiment not chosen! Please include the experiment name as argument."
exp_dir = import_module(sys.argv[1])

assert len(VALUES) == len(VARIABLES), \
    "VALUES and VARIABLES must have the same lenght!"
if len(VALUES) == 0:
    VALUES = [np.array([0])]
    VARIABLES = ['']
var_dict = dict(zip(VARIABLES, VALUES))

# 2. loop over everyting
for run in xrange(RUNS):
    for var in var_dict.keys():
        for value in var_dict[var]:

            # 2.1 load param and experiment files
            params = exp_dir.param

            # 2.2 add experiment specific parameters
            params.get_par()
            if var != '':
                setattr(params.par, var, value)
            params.get_aux()
            params.aux.display = DISPLAY_PROGRESS
            params.aux.experiment_tag = EXPERIMENT_TAG

            # 3. initialize sorn, source, and stats objects
            experiment = exp_dir.experiment.Experiment(params)
            sorn = Sorn(params, experiment.inputsource)
            stats = Stats(experiment.stats_tostore, params)

            # 4. run one experiment once
            if var != '':
                for elem in VARIABLES:
                    print elem, '=', getattr(params.par, elem),
                    print '-- Exp.', run + 1
            else:
                print 'Exp.', run + 1
            experiment.run(sorn, stats)

            # 5. save initial sorn parameters and stats objects
            backup_pickle(experiment, stats)
