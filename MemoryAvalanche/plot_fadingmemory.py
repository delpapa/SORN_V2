""" Performance vs. past time step"""

import os

import cPickle as pickle
import numpy as np
import matplotlib.pylab as plt

# parameters to include in the plot
N_PAR = 200                                    # network size
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

# filter N
n_index = np.where(n_list == N_PAR)[0].tolist()
n_list = n_list[n_index]
a_list = a_list[n_index]
sigma_list = sigma_list[n_index]
performance_list = performance_list[n_index]
assert len(n_list) > 0, 'Currently no experiment data for chosen network size'

# 3. plot memory vs. network size
for alphabet in np.unique(a_list):
    n_list_reduced = n_list[np.where(a_list == alphabet)]
    a_list_reduced = a_list[np.where(a_list == alphabet)]
    sigma_list_reduced = sigma_list[np.where(a_list == alphabet)]
    performance_list_reduced = performance_list[np.where(a_list == alphabet)]
    plt.plot(performance_list_reduced.mean(0),
             '-o',
             label=r'$A = %d$' %int(alphabet))

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
        plt.savefig(plots_dir+'performance_vs_tpast_N'+str(N_PAR)+'.pdf',
                    format='pdf')
plt.show()
