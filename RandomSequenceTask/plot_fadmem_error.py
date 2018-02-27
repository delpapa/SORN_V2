"""
Plot error as a function of T = t - tpast
Curves for each alphabet size A
Network size fixed in N=200
Read experiments from RST_perfXtplast/
Save plots at RST_perfXtplast/
This was used as Fig. 2A in the tentative ESANN abstract.
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
A_PAR = np.array([10, 20, 100, 200])               # input alphabet sizes
SAVE_PLOT = True

from matplotlib import cm
start = 0.
stop = 1.
number_of_lines= 10
cm_subsection = np.linspace(start, stop, number_of_lines)

colors = [ cm.Blues(x) for x in cm_subsection ]

################################################################################
#                             Make plot                                        #
################################################################################

# 0. build figures
fig = plt.figure(1, figsize=(5.3, 5))

# 1. load performances and experiment parameters
print '\nCalculating memory for the Random Sequence Task...'
experiment_tag = '_perfXtplast'
experiment_folder = 'RandomSequenceTask' + experiment_tag
experiment_path = 'backup/' + experiment_folder + '/'
experiment_n = len(os.listdir(experiment_path))

performance_list = np.zeros((experiment_n, 20))
t_list = np.zeros(experiment_n)
a_list = np.zeros(experiment_n)
for exp, exp_name in enumerate(os.listdir(experiment_path)):
    exp_n = [int(s) for s in exp_name.split('_') if s.isdigit()]
    stats = pickle.load(open(experiment_path+exp_name+'/stats.p', 'rb'))
    params = pickle.load(open(experiment_path+exp_name+'/init_params.p', 'rb'))
    performance_list[exp] = stats.performance
    a_list[exp] = int(params.A)
    t_list[exp] = int(params.steps_plastic)

# 2. filter plot data points
# filter A
if 'A_PAR' in globals():
    a_index = []
    for a in A_PAR:
        a_index.extend(np.where(a_list == a)[0].tolist())
    a_list = a_list[a_index]
    t_list = t_list[a_index]
    performance_list = performance_list[a_index]

# 3. plot memory vs. network size
memory = np.zeros( (len(np.unique(a_list)), len(np.unique(t_list))) )

# A loop
for j, a in enumerate(np.unique(a_list)):

    a_list_part = a_list[np.where(a_list == a)]
    t_list_part = t_list[np.where(a_list == a)]
    performance_list_part = performance_list[np.where(a_list == a)]
    for i, tplast in enumerate(np.unique(t_list_part)):
        perf_array = performance_list_part[np.where(t_list_part == tplast)].mean(0)

    if a in A_PAR:
        if a == 10:
            plt.plot(1 - perf_array,
                color=colors[4],
                linestyle='-',
                linewidth=3,
                label=r'$%d$' %a)

        if a == 20:
            plt.plot(1 - perf_array,
                color=colors[5],
                linestyle='-',
                linewidth=3,
                label=r'$%d$' %a)

        if a == 100:
            plt.plot(1 - perf_array,
                color=colors[8],
                linestyle='-',
                linewidth=3,
                label=r'$%d$' %a)

        if a == 200:
            plt.plot(1 - perf_array,
                color=colors[9],
                linestyle='-',
                linewidth=3,
                label=r'$%d$' %a)

# 4. adjust figure parameters and save
fig_lettersize = 15
# plt.title('Random Sequence Task - Convergence')
plt.axhline(0.10, linestyle='--', color='gray')
leg = plt.legend(loc='best', title='Alphabet size', frameon=False, fontsize=fig_lettersize)
leg.get_title().set_fontsize(fig_lettersize)
plt.xlabel(r'$t_{\rm p}$', fontsize=fig_lettersize)
plt.ylabel('Error', fontsize=fig_lettersize)
plt.xlim([0, 10])
plt.ylim([0, 1])
plt.xticks(
    [0, 2, 4, 6, 8, 10],
    ['$0$', '$2$', '$4$', '$6$', '$8$', '$10$'],
    size=fig_lettersize,
    )
plt.yticks(
    [0, 0.1, 0.5, 1.0],
    ['$0\%$', '$10\%$', '$50\%$', '$100\%$'],
    size=fig_lettersize,
)
plt.tight_layout()

if SAVE_PLOT:
    plots_dir = 'plots/ms/'
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
    plt.savefig(plots_dir+'fadmem_error.pdf', format='pdf')
plt.show()
