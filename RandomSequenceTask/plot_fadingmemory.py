""" Performance vs. past time step"""

import os
import sys; sys.path.append('.')
import  pickle

import numpy as np
import matplotlib.pylab as plt

from common.stats import Stats

# parameters to include in the plot
NETWORK_SIZE = 1600
A_PAR = np.array([10, 20, 100, 200])    # input alphabet sizes
SAVE_PLOT = True

################################################################################
#                             Make plot                                        #
################################################################################

# 0. build figures
fig = plt.figure(1, figsize=(6, 5))

# 1. load performances and experiment parameters
print('\nCalculating memory for the Random Sequence Task...')

experiment_folder = 'RandomSequenceTask_FM/'
experiment_path = 'backup/{}/'.format(experiment_folder)
experiment_n = len(os.listdir(experiment_path))

performance_list = []
a_list = []
for exp, exp_name in enumerate(os.listdir(experiment_path)):

    exp_n = [int(s) for s in exp_name.split('_') if s.isdigit()]
    perf = pickle.load(open('{}/stats.p'.format(experiment_path+exp_name), 'rb'))
    params = pickle.load(open('{}/init_params.p'.format(experiment_path+exp_name), 'rb'))
    if params.N_e == NETWORK_SIZE:
        performance_list.append(perf.performance)
        a_list.append(params.A)
performance = np.array(performance_list)

for a in np.unique(a_list):
    if a in A_PAR:
        _perf = performance[np.where(a_list == a)]
        plt.errorbar(np.arange(20), 1 -_perf.mean(0),
                        # yerr=[performance.std(0), performance.std(0)],
                        fmt='--o',
                        label=a)

# 4. adjust figure parameters and save
fig_lettersize = 17
plt.legend(loc='best', frameon=False, title='Alphabet size', fontsize=fig_lettersize, title_fontsize=fig_lettersize)
plt.xlabel(r'$t_{p}$', fontsize=fig_lettersize)
plt.ylabel('Error', fontsize=fig_lettersize)
plt.xlim([0, 20])
plt.ylim([0, 1])
plt.xticks([0, 5, 10, 15, 20], fontsize=fig_lettersize-2)
plt.yticks([0, 0.5, 1], ['$0\%$', '$50\%$', '$100\%$'], fontsize=fig_lettersize-2)
plt.tight_layout()

if SAVE_PLOT:
    plots_dir = 'plots/RandomSequenceTask/'
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
    plt.savefig('{}FM_N{}.pdf'.format(plots_dir, NETWORK_SIZE),
                     format='pdf')
plt.show()
