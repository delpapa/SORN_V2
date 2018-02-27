""" Performance vs. past time step"""

import os

import cPickle as pickle
import numpy as np
import matplotlib.pylab as plt

# parameters to include in the plot
N_PAR = 200                                    # network size
# A_PAR = np.array([4, 10, 20, 30, 40, 50])    # input alphabet sizes
SAVE_PLOT = True

################################################################################
#                             Make plot                                        #
################################################################################

# 0. build figures
fig = plt.figure(1, figsize=(6, 5))

# 1. load performances and experiment parameters
print '\nCalculating memory for the Random Sequence Task...'

for experiment_tag in ['N200_LC']:
    experiment_folder = 'RandomSequenceTask_' + experiment_tag
    experiment_path = 'backup/' + experiment_folder + '/'
    experiment_n = len(os.listdir(experiment_path))

    performance_list = []
    l_list = []
    for exp, exp_name in enumerate(os.listdir(experiment_path)):

        exp_n = [int(s) for s in exp_name.split('_') if s.isdigit()]
        perf = pickle.load(open(experiment_path+exp_name+'/stats.p', 'rb'))
        params = pickle.load(open(experiment_path+exp_name+'/init_params.p', 'rb'))
        performance_list.append(perf.performance)
        l_list.append(params.L)

    performance = np.array(performance_list)
    L = np.array(l_list)

    for l in [200, 500, 1000, 4000]:
        plt.plot(np.arange(20), performance[np.where(L == l)].mean(0),
                 '--o',
                 label=str(l))

# 4. adjust figure parameters and save
fig_lettersize = 12
plt.title('Fading Memory')
plt.legend(loc='best', title='L', frameon=False)
plt.xlabel(r'$t_{\rm past}$', fontsize=fig_lettersize)
plt.ylabel('Performance', fontsize=fig_lettersize)
plt.xlim([0, 20])
plt.xticks([0, 5, 10, 15, 20])

if SAVE_PLOT:
    plots_dir = 'plots/RandomSequenceTask_FadingMemory/'
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
    plt.savefig(plots_dir+'fadingmemory_LC.pdf',
                     format='pdf')
plt.show()
