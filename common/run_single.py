import os
import sys
sys.path.insert(0, '')
from importlib import import_module

import numpy as np
import matplotlib.pyplot as plt

from common.sorn import Sorn
from common.stats import Stats
from utils import backup_pickle

import CountingTask as exp_dir                 # experiment directory

################################################################################
#                              SORN simulation                                 #
################################################################################

# 1. load param and experiment files
param_file = exp_dir.param
experiment_file = exp_dir.experiment

# 2. add experiment specific parameters
param_file.c.display = True                      # display progress
param_file.c.experimentmark = ''                 # to mark different experiments

# 3. initialize sorn, source, and stats objects
experiment = experiment_file.Experiment(param_file.c)
(source, stats_tosave) = experiment.start()
sorn = Sorn(param_file.c,source)
stats = Stats(stats_tosave, sorn.params)

# 4. run one experiment once and calculate performance
experiment.run(sorn, stats)

# 5. save sorn and stats objects
backup_pickle(sorn.params, stats, save_stats = False)
