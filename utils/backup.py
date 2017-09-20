"""Backup handler

This script is contains the backup handling functions.
"""

import os
import time
import cPickle as pickle
import shutil
from shutil import ignore_patterns

import numpy as np

def backup_pickle(experiment, stats):
    ''''
    Back up handling function

    Parameters:
        experiment: Experiment object, contains the initial sorn parameters

        stats: bunch of stats stored during the simulation
    '''
    params = experiment.init_params
    results_dir = experiment.results_dir
    files_tosave = experiment.files_tosave
    directory = ('backup/' + results_dir)

    # creates a new directory for storing the results
    # sleeps for a short time to avoid conflicts when running in parallel
    time.sleep(np.random.rand())
    for n_sim in xrange(1, 100):
        final_dir = directory + '_' + str(n_sim) + '/'
        if not os.path.exists(final_dir):
            os.makedirs(final_dir)
            break

    if 'params' in files_tosave:
        with open(final_dir+'init_params.p', 'wb') as f:
            pickle.dump(params, f)

    if 'stats' in files_tosave:
        with open(final_dir+'stats.p', 'wb') as f:
            # TODO: not sure why I cannot remove the store_step method
            # save only the performance
            if 'performance_only' in files_tosave:
                pickle.dump(stats.performance, f)
            else:
                pickle.dump(stats, f)

    if 'scripts' in files_tosave:
        for f in ['utils', 'common', results_dir.split('/')[0]]:
            shutil.copytree(f, final_dir+f,
                            ignore=ignore_patterns('*.pyc', '*.git'))
