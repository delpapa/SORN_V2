"""
Plot relative number of errors and new sentences as a function of the SORN size
"""

import os
import sys; sys.path.append('.')

import pickle
import numpy as np
import matplotlib.pylab as plt

# parameters to include in the plot
SAVE_PLOT = True

################################################################################
#                             Make plot                                        #
################################################################################

# 0. build figures
fig = plt.figure(1, figsize=(7, 6))

# 1. load performances and experiment parameters
try:
    experiment_tag = sys.argv[1]
except:
    raise ValueError('Please specify a valid experiment tag.')
print('\nCalculating percent of wrong and new sentences... ')

experiment_folder = 'GrammarTask_{}'.format(experiment_tag)
experiment_path = 'backup/' + experiment_folder + '/'
experiment_n = len(os.listdir(experiment_path))
percent_new = np.zeros(experiment_n)
percent_error = np.zeros(experiment_n)
n_removed = np.zeros(experiment_n)
net_size = np.zeros(experiment_n)

for exp, exp_name in enumerate(os.listdir(experiment_path)):

    exp_n = [int(s) for s in exp_name.split('_') if s.isdigit()]
    stats = pickle.load(open(experiment_path+exp_name+'/stats.p', 'rb'))
    params = pickle.load(open(experiment_path+exp_name+'/init_params.p', 'rb'))
    percent_new[exp] = float(stats.n_new)/stats.n_output
    percent_error[exp] = float(stats.n_wrong)/stats.n_output
    n_removed[exp] = int(params.n_removed_sentences)
    net_size[exp] = int(exp_name.split('_')[0][1:])


rem = np.unique(n_removed)
new = []
new_std = []
new_up = []
new_down = []
wrong = []
wrong_std = []
wrong_up = []
wrong_down = []

for i, n in enumerate(rem):
    net = np.where(n == n_removed)
    new.append(percent_new[net].mean())
    new_std.append(percent_new[net].std()/1.2)
    new_up.append(np.percentile(percent_new[net], 60))
    new_down.append(np.percentile(percent_new[net], 40))
    wrong.append(percent_error[net].mean())
    wrong_std.append(percent_error[net].std())
    wrong_up.append(np.percentile(percent_error[net], 75))
    wrong_down.append(np.percentile(percent_error[net], 25))

new_color = 'blue'
wrong_color = 'red'

plt.plot(rem, new, '-o',
         color=new_color, alpha=0.5,
         label='new')
plt.plot(rem, wrong, '-o',
         color=wrong_color, alpha=0.5,
         label='incorrect')

plt.fill_between(rem,
                 np.array(new)-np.array(new_std),
                 np.array(new)+np.array(new_std),
                 color=new_color, alpha=0.2)
plt.fill_between(rem,
                 np.array(wrong)-np.array(wrong_std),
                 np.array(wrong)+np.array(wrong_std),
                 color=wrong_color, alpha=0.2)

# 4. adjust figure parameters and save
fig_lettersize = 20
# plt.title('Plasticity effects')
plt.legend(loc='best', frameon=False, fontsize=fig_lettersize)
plt.xlabel(r'excluded sentences (%)', fontsize=fig_lettersize)
plt.ylabel(r'sentences (%)', fontsize=fig_lettersize)
plt.xlim([0, 64])
plt.ylim([0., 0.5])
plt.xticks(
    [16, 32, 48, 64],
    ['$25$', '$50$', '$75$', '$100$'],
    size=fig_lettersize-2,
)
plt.yticks(
    [0., 0.1, 0.2, 0.3, 0.4, 0.5],
    ['$0$', '$10$', '$20$', '$30$', '$40$', '$50$'],
    size=fig_lettersize,
)


if SAVE_PLOT:
    plots_dir = 'plots/GrammarTask/'
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
    plt.savefig('{}removed_sentences.pdf'.format(plots_dir), format='pdf')
plt.show()
