"""Backup handler

This script is contains the backup handling functions.
"""

import os
import time
import pickle
import shutil
from shutil import ignore_patterns

import numpy as np


def backup_pickle(experiment, stats):
    ''''
    Back up handling function.

    Arguments:
    experiment -- Experiment object, contains the initial sorn parameters
    stats -- bunch of stats stored during the simulation
    '''

    params = experiment.init_params
    results_dir = experiment.results_dir
    files_tosave = experiment.files_tosave
    directory = ('backup/{}'.format(results_dir))

    # creates a new directory for storing the results
    # sleeps for a short time to avoid conflicts when running in parallel
    time.sleep(np.random.rand())
    for n_sim in range(1, 1000):
        final_dir = '{}_{}/'.format(directory, str(n_sim))
        if not os.path.exists(final_dir):
            try:
                os.makedirs(final_dir)
                break
            except:
                pass

    if 'params' in files_tosave:
        with open(final_dir+'init_params.p', 'wb') as f:
            pickle.dump(params, f)

    if 'stats' in files_tosave:
        # delete attributes that occupy a lot of memory space
        if hasattr(stats, 'input_index_readout'):
            del stats.input_index_readout
        if hasattr(stats, 'input_readout'):
            del stats.input_readout
        if hasattr(stats, 'raster_readout'):
            del stats.raster_readout
        if hasattr(stats, 't_past'):
            del stats.t_past

        with open(final_dir+'stats.p', 'wb') as f:
            pickle.dump(stats, f)

    if 'scripts' in files_tosave:
        # TODO: this should not need a '_'
        for f in ['utils', 'common', results_dir.split('_')[0]]:
            shutil.copytree(f, final_dir+f,
                            ignore=ignore_patterns('*.pyc', '*.git'))
