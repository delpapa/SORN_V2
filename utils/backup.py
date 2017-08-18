import sys
import os
import time
import cPickle as pickle
import shutil
from shutil import ignore_patterns

import numpy as np

def backup_pickle(params, results_dir, stats,
                  save_params = True,
                  save_stats = True,
                  save_performance_only = True,
                  save_dirs = True):
    ''''
    Back up handling function

    Parameters:
        par: bunch of simulation parameters

        stats: bunch of stats saved during the simulation

        save_params: if True, backup params

        save_stats: if True, backup stats

        save_performance_only: if True, backup the simulation performance

        save_dirs: if True, backup every file used in the simulation
    '''
    directory = ('backup/' + results_dir)

    # creates a new directory for storing the results
    # sleeps for a short time to avoid conflicts when running in parallel
    time.sleep(np.random.rand())
    for n_sim in xrange(1, 1000):
        final_dir = directory + '_' + str(n_sim) + '/'
        if not os.path.exists(final_dir):
            os.makedirs(final_dir)
            break

    if save_params:
        with open(final_dir+'params.p', 'wb') as f:
            pickle.dump(params, f)

    if save_stats:
        with open(final_dir+'stats.p', 'wb') as f:
            pickle.dump(stats, f)

    elif save_performance_only:
        with open(final_dir+'stats.p', 'wb') as f:
            pickle.dump(stats.LG_performance, f)

    if save_dirs:
        for f in ['utils', 'common', results_dir.split('/')[0]]:
            shutil.copytree(f, final_dir+f,
                            ignore=ignore_patterns('*.pyc', '*.git'))


def backup_h5(params, stats,
                  save_params = True,
                  save_stats = True,
                  save_performance_only = True,
                  save_dirs = True):

    # TODO: this should be implemented ASAP
    pass
