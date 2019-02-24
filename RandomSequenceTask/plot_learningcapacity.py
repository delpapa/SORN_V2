import os
import pickle

import sys; sys.path.append('.')
import numpy as np
import matplotlib.pylab as plt

# parameters to include in the plot
SAVE_PLOT = True
################################################################################
#                             Make plot                                        #
################################################################################

# 0. build figures
fig = plt.figure(1, figsize=(6, 6))

# 1. load performances and experiment parameters
try:
    experiment_tag = sys.argv[1]
except:
    raise ValueError('Please specify a valid experiment tag.')

print('\nCalculating learning capacity for the Random Sequence Task...')

experiment_folder = 'RandomSequenceTask_{}'.format(experiment_tag)
experiment_path = 'backup/{}/'.format(experiment_folder)
experiment_n = len(os.listdir(experiment_path))
performance = np.zeros(experiment_n)
l_list = np.zeros(experiment_n, dtype=np.int)
n_list = np.zeros(experiment_n, dtype=np.int)

for exp, exp_name in enumerate(os.listdir(experiment_path)):
    exp_n = [int(s) for s in exp_name.split('_') if s.isdigit()]
    stats = pickle.load(open('{}/stats.p'.format(experiment_path+exp_name), 'rb'))
    params = pickle.load(open('{}/init_params.p'.format(experiment_path+exp_name), 'rb'))
    performance[exp] = stats.performance
    l_list[exp] = int(params.L)
    n_list[exp] = int(params.N_e)

print(n_list)
print(performance)

performance_mean = np.zeros(len(np.unique(l_list)))
for i, l in enumerate(np.unique(l_list)):
    performance_mean[i] = performance[np.where(l_list == l)].mean()
plt.plot(np.unique(l_list), 1-performance_mean, 'o', label=experiment_tag[5:])

# 4. adjust figure parameters and save
fig_lettersize = 15
plt.legend(loc=(0.05, 0.55), frameon=False, fontsize=fig_lettersize, title='Network size')
plt.xlabel(r'$L$', fontsize=fig_lettersize)
plt.ylabel(r'error', fontsize=fig_lettersize)
# plt.xlim([0, 8000])
plt.ylim([0., 0.9])
plt.xscale('log')

if SAVE_PLOT:
    plots_dir = 'plots/RandomSequenceTask/'
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
    plt.savefig('{}LC.pdf'.format(plots_dir), format='pdf')
plt.show()
