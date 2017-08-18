import os
import sys
sys.path.insert(0, '')
from importlib import import_module

import numpy as np
import matplotlib.pyplot as plt

from common.sorn import Sorn
from common.stats import Stats
from utils import backup_pickle

import RandomSequenceTask as exp_dir                 # experiment directory

################################################################################
#                              SORN simulation                                 #
################################################################################

# 1. load param and experiment files
params = exp_dir.param
experiment_file = exp_dir.experiment

# 2. add experiment specific parameters
params.get_aux()
params.aux.display = True
params.aux.experiment_tag = ''

# 3. initialize sorn, source, and stats objects
experiment = experiment_file.Experiment(params.c)
(source, stats_tosave) = experiment.start(params.c)
sorn = Sorn(params.c, source)
stats = Stats(stats_tosave, params.c)

# 4. run one experiment once and calculate performance
experiment.run(sorn, stats)

# 5. save sorn and stats objects
backup_pickle(sorn.params, stats, save_stats = False)
