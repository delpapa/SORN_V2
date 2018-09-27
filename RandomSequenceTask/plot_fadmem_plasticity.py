"""
Plot fading memory as a  function ot TPlast
Curves for different combinations of plastcity mechs.
Network size in [100, 200]
Read experiments from RST_perfXtplast_*
Save plots at RST_perfXtplast_plasticity/
This was used as Fig. 2B in the tentative ESANN abstract.
"""

import os
import sys
sys.path.insert(0, '')

import cPickle as pickle
import numpy as np
import matplotlib.pylab as plt
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d

# parameters to include in the plot
SAVE_PLOT = True

################################################################################
#                             Make plot                                        #
################################################################################

# calculate the fading memory with a threshold m
def fading_memory(perf_data):
    """Calculate how many past time steps are necessary for half performance"""

    m = 0.90 # memory threshold
    memory = np.zeros(len(perf_data))
    for i in xrange(len(perf_data)):
        memory[i] = np.where(perf_data[i] >= m)[0].size

    return memory.mean(), memory.std()

# 0. build figures
fig = plt.figure(1, figsize=(7, 6))

# 1. load performances and experiment parameters
print '\nCalculating memory for the Random Sequence Task...'
experiment_tag = '_perfXtplast_'
experiment_folder = 'RandomSequenceTask' + experiment_tag


for i in ['RR', 'NoIP', 'NoSTDPPrune', 'SORN', 'smallSORN']:

    experiment_path = 'backup/' + experiment_folder + i + '/'
    experiment_n = len(os.listdir(experiment_path))
    performance = np.zeros((experiment_n, 20))
    t_list = np.zeros(experiment_n)

    for exp, exp_name in enumerate(os.listdir(experiment_path)):
        exp_n = [int(s) for s in exp_name.split('_') if s.isdigit()]
        stats = pickle.load(open(experiment_path+exp_name+'/stats.p', 'rb'))
        params = pickle.load(open(experiment_path+exp_name+'/init_params.p', 'rb'))
        performance[exp] = stats.performance
        t_list[exp] = int(params.steps_plastic)

    memory_mean = np.zeros(len(np.unique(t_list)))
    memory_std = np.zeros(len(np.unique(t_list)))

    for j, tplast in enumerate(np.unique(t_list)):
        performance_local = performance[np.where(t_list == tplast)]
        # remove lines that are 0
        performance_local = performance_local[~np.all(performance_local == 0, axis=1)] #remove 0 lines
        memory_mean[j], memory_std[j] = fading_memory(performance_local)

    if i == 'SORN':
        plt.plot(np.unique(t_list), memory_mean,
                 linewidth=3,
                 color='k',
                 label='SORN (STDP + IP)')
    if i == 'RR':
        plt.plot(np.unique(t_list), memory_mean,
                 linewidth=3,
                 label='Random Network')
    if i == 'NoIP':
        plt.plot(np.unique(t_list), memory_mean,
                 linewidth=3,
                 label='STDP')
    # if i == 'NoSTDPPrune':
    #     plt.plot(np.unique(t_list), memory_mean,
    #              linewidth=1,
    #              color='k',
    #              linestyle='-.',
    #              label='only IP')
    if i == 'smallSORN':
        plt.plot(np.unique(t_list), memory_mean,
                 linewidth=3,
                 color='gray',
                 label='SORN (STDP + IP,  $N^E = 100$)')



# 4. adjust figure parameters and save
fig_lettersize = 15
# plt.title('Plasticity effects')
plt.legend(loc=(0.21, 0.001), frameon=False, fontsize=fig_lettersize)
plt.xlabel(r'$T_{\rm plasticity}$', fontsize=fig_lettersize)
plt.ylabel(r'$M_{ \tau }$', fontsize=fig_lettersize)
plt.xlim([0, 30000])
plt.ylim([0.5, 3.5])
plt.xticks(
    [0, 10000, 20000, 30000],
    ['0', '$1\cdot10^4$', '$2\cdot10^4$', '$3\cdot10^4$'],
    size=fig_lettersize,
    )
plt.yticks(
    [1, 2, 3],
    ['$1$', '$2$', '$3$'],
    size=fig_lettersize,
    )
plt.tight_layout()

if SAVE_PLOT:
    plots_dir = 'plots/ms/'
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
    plt.savefig(plots_dir+'fadmem_plasticity.pdf', format='pdf')
plt.show()
