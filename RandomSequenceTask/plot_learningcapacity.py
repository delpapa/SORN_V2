"""
Plot fading memory as a  function ot TPlast
Curves for different combinations of plastcity mechs.
Network size in [100, 200]
Read experiments from RST_perfXtplast_*
Save plots at RST_perfXtplast_plasticity/
This was used as Fig. 2B in the tentative ESANN abstract.
"""

import os

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

# 0. build figures
fig = plt.figure(1, figsize=(6, 6))

# 1. load performances and experiment parameters
print '\nCalculating learning capacity for the Random Sequence Task...'

# for experiment_tag in ['_LC_N200_logn', '_LC_N200', '_LC_N200_RR']:
for experiment_tag in ['_LC_N100', '_LC_N200', '_LC_N300', '_LC_N400', '_LC_N500', '_LC_N800']:
    experiment_folder = 'RandomSequenceTask' + experiment_tag
    experiment_path = 'backup/' + experiment_folder + '/'
    experiment_n = len(os.listdir(experiment_path))
    performance = np.zeros(experiment_n)
    l_list = np.zeros(experiment_n)

    for exp, exp_name in enumerate(os.listdir(experiment_path)):
        exp_n = [int(s) for s in exp_name.split('_') if s.isdigit()]
        stats = pickle.load(open(experiment_path+exp_name+'/stats.p', 'rb'))
        params = pickle.load(open(experiment_path+exp_name+'/init_params.p', 'rb'))
        performance[exp] = stats.performance
        l_list[exp] = int(params.L)

        performance_mean = np.zeros(len(np.unique(l_list)))
        performance_low = np.zeros(len(np.unique(l_list)))
        performance_high = np.zeros(len(np.unique(l_list)))
    for i, l in enumerate(np.unique(l_list)):

        performance_mean[i] = performance[np.where(l_list == l)].mean()
        performance_low[i]= np.percentile(performance[np.where(l_list == l)], 33)
        performance_high[i]= np.percentile(performance[np.where(l_list == l)], 66)


    # plt.errorbar(np.unique(l_list), 1-performance_mean,
    #              yerr=[1-performance_high, 1-performance_low],
    #              fmt='--o',
    #              label=experiment_tag[4:])
    plt.plot(np.unique(l_list), 1-performance_mean, '--o', label=experiment_tag[4:])


# 4. adjust figure parameters and save
fig_lettersize = 15
# plt.title('Plasticity effects')
plt.axhline(0.01, color='gray', linestyle='--')
plt.legend(loc=(0.05, 0.55), frameon=False, fontsize=fig_lettersize)
plt.xlabel(r'$L$', fontsize=fig_lettersize)
plt.ylabel(r'error', fontsize=fig_lettersize)
# plt.xlim([0, 8000])
plt.ylim([0., 0.9])
plt.xscale('log')
# plt.xticks(
#     [0, 10000, 20000, 30000],
#     ['0', '$1\cdot10^4$', '$2\cdot10^4$', '$3\cdot10^4$'],
#     size=fig_lettersize,
#     )
# plt.yticks(
#     [1, 2, 3],
#     ['$1$', '$2$', '$3$'],
#     size=fig_lettersize,
#     )

if SAVE_PLOT:
    plots_dir = 'plots/RandomSequenceTask_LC_logn/'
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
    plt.savefig(plots_dir+'leaarningcapacity.pdf', format='pdf')
plt.show()
