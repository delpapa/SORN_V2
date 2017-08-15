import sys
import os
import time
import cPickle as pickle
import shutil
from shutil import ignore_patterns

import numpy as np

def backup_pickle(params, stats,
                  save_params = True,
                  save_stats = True,
                  save_performance_only = True,
                  save_dirs = True):
    ''''
    Back up handling function

    Parameters:
        params: bunch of simulation parameters

        stats: bunch of stats saved during the simulation

        save_params: if True, backup params

        save_stats: if True, backup stats

        save_performance_only: if True, backup the simulation performance

        save_dirs: if True, backup every file used in the simulation
    '''
    if params.display:
        sys.stdout.write('Saving sorn and stats...')

    if params.experiment_name == 'CountingTask':
        directory = ('backup/' + params.experiment_name + params.experimentmark
                     + '/N' + str(params.N_e) + '_L' + str(params.L))
    else:
        directory = ('backup/' + params.experiment_name + params.experimentmark
                     + '/N' + str(params.N_e) + '_L' + str(params.L)
                     + '_A'+str(params.A))

    # creates a new directory for storing the results
    # sleeps for a short time to avoid conflicts when using a cluster
    time.sleep(np.random.rand())
    for n_sim in xrange(1, 1000):
        final_dir = directory + '_' + str(n_sim) + '/'
        if not os.path.exists(final_dir):
            os.makedirs(final_dir)
            break

    if save_params:
        with open(final_dir+'sorn.p', 'wb') as f:
            pickle.dump(params, f)

    if save_stats:
        with open(final_dir+'stats.p', 'wb') as f:
            pickle.dump(stats, f)

    elif save_performance_only:
        with open(final_dir+'stats.p', 'wb') as f:
            pickle.dump(stats.LG_performance, f)

    if save_dirs:
        for f in ['utils', 'common', params.experiment_name]:
            shutil.copytree(f, final_dir+f,
                            ignore=ignore_patterns('*.pyc', '*.git'))

    if params.display:
        sys.stdout.write('done \n\n')


def backup_h5(params, stats,
                  save_params = True,
                  save_stats = True,
                  save_performance_only = True,
                  save_dirs = True):

    # TODO: this should be implemented ASAP
    pass
