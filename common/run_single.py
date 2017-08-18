import os
import sys
sys.path.insert(0, '')
from importlib import import_module

import numpy as np
import matplotlib.pyplot as plt

from common.sorn import Sorn
from common.stats import Stats
from utils import backup_pickle

################################################################################
#                              SORN simulation                                 #
################################################################################

# 1. load param and experiment files
exp_dir = import_module(sys.argv[1])
params = exp_dir.param
experiment_file = exp_dir.experiment

# 2. add experiment specific parameters
params.get_par()
params.get_aux()
params.aux.display = True
params.aux.experiment_tag = ''

# 3. initialize sorn, source, and stats objects
experiment = experiment_file.Experiment(params)
experiment.start()
sorn = Sorn(params, experiment.inputsource)
stats = Stats(experiment.stats_tosave, sorn.params)

# 4. run one experiment once and calculate performance
experiment.run(sorn, stats)

# 5. save initial sorn parameters and stats objects
backup_pickle(experiment.init_params,
              experiment.results_dir,
              experiment.files_tosave,
              stats)
