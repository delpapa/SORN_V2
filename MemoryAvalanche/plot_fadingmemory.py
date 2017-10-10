""" Performance vs. past time step"""

import os

import cPickle as pickle
import numpy as np
import matplotlib.pylab as plt

# parameters to include in the plot
N_PAR = np.array([200])
SIGMA_PAR = np.array([0., 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1])                                    # noise level
A_PAR = np.array([20])    # input alphabet sizes
SAVE_PLOT = True

################################################################################
#                             Make plot                                        #
################################################################################

# 0. build figures
fig = plt.figure(1, figsize=(6, 5))

# 1. load performances and experiment parameters
print '\nCalculating memory for the Random Sequence Task...'
experiment_tag = ''
experiment_folder = 'MemoryAvalanche_PZ' + experiment_tag
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
    n_list_reduced = n_list[np.where(sigma_list == noise_level)]
    a_list_reduced = a_list[np.where(sigma_list == noise_level)]
    sigma_list_reduced = sigma_list[np.where(sigma_list == noise_level)]
    performance_list_reduced = performance_list[np.where(sigma_list == noise_level)]
    plt.plot(performance_list_reduced.mean(0),
             '-o',
             label=r'$\sigma^2 = %.3f$' %float(noise_level))

# 4. adjust figure parameters and save
fig_lettersize = 12
plt.title('Criticality - Fading Memory')
plt.legend(loc='best')
plt.xlabel(r'$t_{\rm past}$', fontsize=fig_lettersize)
plt.ylabel('Performance', fontsize=fig_lettersize)
plt.xlim([0, 20])
plt.xticks([0, 5, 10, 15, 20])

if SAVE_PLOT:
    plots_dir = 'plots/'+experiment_folder+'/'
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
    plt.savefig(plots_dir+'performance_vs_tpast.pdf', format='pdf')
plt.show()
