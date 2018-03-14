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
print '\nCalculating percent of wrong and new sentences... '

experiment_tag = '_FDT-16'
experiment_folder = 'LanguageTask' + experiment_tag
experiment_path = 'backup/' + experiment_folder + '/'
experiment_n = len(os.listdir(experiment_path))
percent_new = np.zeros(experiment_n)
percent_error = np.zeros(experiment_n)
net_size = np.zeros(experiment_n)

for exp, exp_name in enumerate(os.listdir(experiment_path)):

    exp_n = [int(s) for s in exp_name.split('_') if s.isdigit()]
    stats = pickle.load(open(experiment_path+exp_name+'/stats.p', 'rb'))
    percent_new[exp] = float(stats.n_new)/stats.n_output_sentences
    percent_error[exp] = float(stats.n_wrong)/stats.n_output_sentences
    net_size[exp] = int(exp_name.split('_')[0][1:])

N_e = np.unique(net_size)
new = []
new_std = []
new_up = []
new_down = []
wrong = []
wrong_std = []
wrong_up = []
wrong_down = []
for i, n in enumerate(N_e):
    net = np.where(n == net_size)
    new.append(percent_new[net].mean())
    new_std.append(percent_new[net].std())
    new_up.append(np.percentile(percent_new[net], 75))
    new_down.append(np.percentile(percent_new[net], 25))
    wrong.append(percent_error[net].mean())
    wrong_std.append(percent_error[net].std())
    wrong_up.append(np.percentile(percent_error[net], 75))
    wrong_down.append(np.percentile(percent_error[net], 25))
    # print n, percent_new[net], percent_error[net]

plt.plot(N_e, new, color='blue', alpha=0.5, label='New')
plt.plot(N_e, wrong, color='red', alpha=0.5, label='Wrong')
plt.fill_between(N_e,
                 np.array(new)-np.array(new_std),
                 np.array(new)+np.array(new_std),
                 color='blue', alpha=0.2)
plt.fill_between(N_e,
                 np.array(wrong)-np.array(wrong_std),
                 np.array(wrong)+np.array(wrong_std),
                 color='red', alpha=0.2)

# 4. adjust figure parameters and save
fig_lettersize = 15
# plt.title('Plasticity effects')
plt.legend(loc='best', frameon=False, fontsize=fig_lettersize)
plt.xlabel(r'Network size', fontsize=fig_lettersize)
plt.ylabel(r'# sentences (%)', fontsize=fig_lettersize)
# plt.xlim([0, 8000])
plt.ylim([0., 0.4])
# plt.xticks(
#     [0, 10000, 20000, 30000],
#     ['0', '$1\cdot10^4$', '$2\cdot10^4$', '$3\cdot10^4$'],
#     size=fig_lettersize,
#     )
plt.yticks(
    [0., 0.1, 0.2, 0.3, 0.4],
    ['$0$', '$10$', '$20$', '$30$', '$40$'],
    size=fig_lettersize,
    )

if SAVE_PLOT:
    plots_dir = 'plots/' + experiment_folder + '/'
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
    plt.savefig(plots_dir+'error_and_new.pdf', format='pdf')
plt.show()
