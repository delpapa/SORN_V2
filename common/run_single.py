"""This script runs a single sorn simulation for the experiment given as an
argument with the specified parameters and experiment instructions"""

import sys
from importlib import import_module

from common.sorn import Sorn
from common.stats import Stats
from utils import backup_pickle

################################################################################
#                              SORN simulation                                 #
################################################################################

# 1. load param file
sys.path.insert(0, '')
exp_dir = import_module(sys.argv[1])
params = exp_dir.param

# 2. add experiment specific parameters
params.get_par()
params.get_aux()
params.aux.display = True
params.aux.experiment_tag = ''

# 3. initialize experiment, sorn, and stats objects
# PS.the experiment class keeps a copy of the initial sorn main parameters
experiment = exp_dir.experiment.Experiment(params)
sorn = Sorn(params, experiment.inputsource)
stats = Stats(experiment.stats_tostore, params)

# 4. run one experiment once
experiment.run(sorn, stats)

# 5. save initial sorn parameters and stats objects
backup_pickle(experiment, stats)
