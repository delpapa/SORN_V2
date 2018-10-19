"""
Plot relative number of errors and new sentences as a function of the SORN size
"""

import os
import sys
sys.path.insert(0, '')

import cPickle as pickle
import numpy as np
import matplotlib.pylab as plt

import common.stats


# parameters to include in the plot
SAVE_PLOT = True

################################################################################
#                             Make plot                                        #
################################################################################

# 0. build figures
fig = plt.figure(1, figsize=(7, 6))

# 1. load performances and experiment parameters
print '\nCalculating percent of wrong sentences... '

for experiment_tag in ['error_and_new', 'eFDT', 'sp']:
    experiment_folder = 'LanguageTask_' + experiment_tag
    experiment_path = 'backup/' + experiment_folder + '/'
    experiment_n = len(os.listdir(experiment_path))
    percent_error = np.zeros(experiment_n)
    net_size = np.zeros(experiment_n)

    for exp, exp_name in enumerate(os.listdir(experiment_path)):

        exp_n = [int(s) for s in exp_name.split('_') if s.isdigit()]
        stats = pickle.load(open(experiment_path+exp_name+'/stats.p', 'rb'))
        percent_error[exp] = float(stats.n_wrong)/stats.n_output
        net_size[exp] = int(exp_name.split('_')[0][1:])

    N_e = np.unique(net_size)
    wrong = []
    wrong_std = []
    wrong_up = []
    wrong_down = []
    for i, n in enumerate(N_e):
        net = np.where(n == net_size)
        wrong.append(percent_error[net].mean())
        wrong_std.append(percent_error[net].std())
        wrong_up.append(np.percentile(percent_error[net], 75))
        wrong_down.append(np.percentile(percent_error[net], 25))

    if experiment_tag == 'error_and_new':
        plot_label = 'FDT'
        plot_color = 'r'
        plot_style = '-'
    elif experiment_tag == 'eFDT':
        plot_label = 'eFDT'
        plot_color = 'r'
        plot_style = '--'
    elif experiment_tag == 'sp':
        plot_label = 'Sing./Plur.'
        plot_color = 'orange'
        plot_style = '-'

    plt.plot(N_e, wrong, plot_style,
             alpha=0.5,
             color=plot_color,
             label=plot_label)

    plt.fill_between(N_e,
                     np.array(wrong)-np.array(wrong_std),
                     np.array(wrong)+np.array(wrong_std),
                     color=plot_color,
                     alpha=0.2)

# 4. adjust figure parameters and save
fig_lettersize = 20
# plt.title('Plasticity effects')
plt.legend(loc='best', frameon=False, fontsize=fig_lettersize)
plt.xlabel(r'$N^{\rm E}$', fontsize=fig_lettersize)
plt.ylabel(r' incorrect sentences (%)', fontsize=fig_lettersize)
plt.xlim([100, 1500])
plt.ylim([0., 0.6])
plt.xticks(
    [200, 500, 800, 1100, 1400, 1700],
    ['$200$',  '$500$',  '$800$', '$1100$', '$1400$', '$1700$'],
    size=fig_lettersize,
    )
plt.yticks(
    [0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
    ['$0$', '$10$', '$20$', '$30$', '$40$', '$50$', '$60$'],
    size=fig_lettersize,
    )

if SAVE_PLOT:
    plots_dir = 'plots/' + experiment_folder + '/'
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
    plt.savefig(plots_dir+'dicts_error.pdf', format='pdf')
plt.show()
