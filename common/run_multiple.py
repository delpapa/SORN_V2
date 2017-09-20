"""This script runs multiple sorn simulations for the experiment given as an
argument with the specified parameters and experiment instructions. parameters
can be varied here, using 'variables' and 'values'"""

import sys
from importlib import import_module

import numpy as np

from common.sorn import Sorn
from common.stats import Stats
from utils import backup_pickle

# variables and values to run: variables must have the same name as in param.py
# these values always overwrite the values in that file
VARIABLES = [
    'N_e',
]
VALUES = [
    np.array([200]),
]

# number of repetitions of each experiment (for statistics)
RUNS = 10

# experiment parameters
DISPLAY_PROGRESS = False                         # display progress bar
EXPERIMENT_TAG = ''                              # to mark experiment

################################################################################
#                              SORN simulation                                 #
################################################################################

# 1. import module and create a dictionary with the variables and values to run
sys.path.insert(0, '')
exp_dir = import_module(sys.argv[1])
import ipdb; ipdb.set_trace()
var_dict = dict(zip(VARIABLES, VALUES))

# 2. loop over everyting
for run in xrange(RUNS):
    for var in var_dict.keys():
        for value in var_dict[var]:


            # 2.1 load param and experiment files
            params = exp_dir.param

            # 2.2 add experiment specific parameters
            params.get_par()
            setattr(params.par, var, value)
            params.get_aux()
            params.aux.display = DISPLAY_PROGRESS
            params.aux.experiment_tag = EXPERIMENT_TAG

            # 3. initialize sorn, source, and stats objects
            experiment = exp_dir.experiment.Experiment(params)
            sorn = Sorn(params, experiment.inputsource)
            stats = Stats(experiment.stats_tostore, params)

            # 4. run one experiment once
            for elem in VARIABLES:
                print elem, '=', getattr(params.par, elem),
            print '-- Exp.', run + 1
            experiment.run(sorn, stats)

            # 5. save initial sorn parameters and stats objects
            backup_pickle(experiment, stats)
