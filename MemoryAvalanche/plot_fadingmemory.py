""" Performance vs. past time step"""

import os
import sys
sys.path.insert(0, '')

import cPickle as pickle
import numpy as np
import matplotlib.pylab as plt
from matplotlib import cm

# parameters to include in the plot
N_PAR = np.array([200])
SIGMA_PAR = np.array([0., 0.05, 0.5, 5])                                    # noise level
A_PAR = np.array([20])    # input alphabet sizes
SAVE_PLOT = True

################################################################################
#                             Make plot                                        #
################################################################################

# 0. build figures
fig = plt.figure(1, figsize=(5.3, 5))

# 1. load performances and experiment parameters
print '\nCalculating memory for the Random Sequence Task...'
experiment_tag = '_PZ'
experiment_folder = 'MemoryAvalanche' + experiment_tag
experiment_path = 'backup/' + experiment_folder + '/'
experiment_n = len(os.listdir(experiment_path))

performance_list = np.zeros((experiment_n, 20))
n_list = np.zeros(experiment_n)
a_list = np.zeros(experiment_n)
l_list = np.zeros(experiment_n)
sigma_list = np.zeros(experiment_n)
for exp, exp_name in enumerate(os.listdir(experiment_path)):
    N, L, A, sigma, _ = [s[1:] for s in exp_name.split('_')]
    exp_n = [int(s) for s in exp_name.split('_') if s.isdigit()]
    print exp
    stats = pickle.load(open(experiment_path+exp_name+'/stats.p', 'rb'))
    performance_list[exp] = stats.performance
    n_list[exp] = int(N)
    a_list[exp] = int(A)
    l_list[exp] = int(L)
    sigma_list[exp] = float(sigma)

# 2. filter plot data points
# filter A
if 'A_PAR' in globals():
    a_index = []
    for a in A_PAR:
        a_index.extend(np.where(a_list == a)[0].tolist())
    a_list = a_list[a_index]
    n_list = n_list[a_index]
    sigma_list = sigma_list[a_index]
    performance_list = performance_list[a_index]

# filter SIGMA
if 'SIGMA_PAR' in globals():
    sigma_index = []
    for sigma in SIGMA_PAR:
        sigma_index.extend(np.where(sigma_list == sigma)[0].tolist())
    n_list = n_list[sigma_index]
    a_list = a_list[sigma_index]
    sigma_list = sigma_list[sigma_index]
    performance_list = performance_list[sigma_index]
assert len(n_list) > 0, 'Currently no experiment data for chosen noise level'

# 3. plot memory vs. network size
for noise_level in np.unique(sigma_list):
    print noise_level
    n_list_reduced = n_list[np.where(sigma_list == noise_level)]
    a_list_reduced = a_list[np.where(sigma_list == noise_level)]
    sigma_list_reduced = sigma_list[np.where(sigma_list == noise_level)]
    performance_list_reduced = performance_list[np.where(sigma_list == noise_level)]

    perf_mean = performance_list_reduced.mean(0)
    perf_err = performance_list_reduced.std(0)

    if float(noise_level) == 0:
        plt.plot(perf_mean,
                '-',
                color='darkcyan',
                linewidth=1.5,
                label=r'low')
        plt.fill_between(range(len(perf_mean)),
                 perf_mean-perf_err,
                 perf_mean+perf_err,
                 color='darkcyan',
                 alpha = 0.5)
    elif float(noise_level) == 0.5:
        plt.plot(perf_mean,
                '-',
                color='k',
                linewidth=2.0,
                label=r'medium')
        plt.fill_between(range(len(perf_mean)),
                 perf_mean-perf_err,
                 perf_mean+perf_err,
                 color='k',
                 alpha = 0.5)
    elif float(noise_level) == 5.0:
        plt.plot(perf_mean,
                '-',
                color='red',
                linewidth=1.5,
                label=r'high')
        plt.fill_between(range(len(perf_mean)),
                perf_mean-perf_err,
                perf_mean+perf_err,
                color='red',
                alpha = 0.5)
#plt.axhline(0.10, linestyle='--', color='gray')

# 4. adjust figure parameters and save
fig_lettersize = 15
leg = plt.legend(loc='best', title='Noise level', frameon=False, fontsize=fig_lettersize)
leg.get_title().set_fontsize(fig_lettersize)
plt.xlabel('$t_p$', fontsize=fig_lettersize)
plt.ylabel('Error', fontsize=fig_lettersize)
#plt.yscale('log')
plt.xlim([0, 6])
plt.ylim([1, 0.0])
plt.yticks([1, 0.5, 0.], ['$0\%$', '$50\%$', '$100\%$'], fontsize=fig_lettersize)
plt.xticks([0, 1, 2, 3, 4, 5, 6], fontsize=fig_lettersize)
plt.tight_layout()

if SAVE_PLOT:
    plots_dir = 'plots/'+experiment_folder+'/'
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
    plt.savefig(plots_dir+'performance_vs_tpast.pdf', format='pdf')
plt.show()
