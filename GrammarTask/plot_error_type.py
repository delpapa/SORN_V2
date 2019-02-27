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
print('\nCalculating percent of error types... ')

experiment_tag = 'sp'
experiment_folder = 'GrammarTask_' + experiment_tag
experiment_path = 'backup/' + experiment_folder + '/'
experiment_n = len(os.listdir(experiment_path))
gram_error = np.zeros(experiment_n)
sema_error = np.zeros(experiment_n)
other_error = np.zeros(experiment_n)
net_size = np.zeros(experiment_n)

for exp, exp_name in enumerate(os.listdir(experiment_path)):

    exp_n = [int(s) for s in exp_name.split('_') if s.isdigit()]
    stats = pickle.load(open(experiment_path+exp_name+'/stats.p', 'rb'))
    gram_error[exp] = float(stats.n_gram)/stats.n_output
    sema_error[exp] = float(stats.n_sema)/stats.n_output
    other_error[exp] = float(stats.n_others)/stats.n_output
    net_size[exp] = int(exp_name.split('_')[0][1:])

N_e = np.unique(net_size)
others = []
others_std = []
others_up = []
others_down = []
gram = []
gram_std = []
gram_up = []
gram_down = []
sema = []
sema_std = []
sema_up = []
sema_down = []
for i, n in enumerate(N_e):
    net = np.where(n == net_size)

    others.append(other_error[net].mean())
    others_std.append(other_error[net].std())
    others_up.append(np.percentile(other_error[net], 75))
    others_down.append(np.percentile(other_error[net], 25))

    gram.append(gram_error[net].mean())
    gram_std.append(gram_error[net].std())
    gram_up.append(np.percentile(gram_error[net], 75))
    gram_down.append(np.percentile(gram_error[net], 25))

    sema.append(sema_error[net].mean())
    sema_std.append(sema_error[net].std())
    sema_up.append(np.percentile(sema_error[net], 75))
    sema_down.append(np.percentile(sema_error[net], 25))

plt.plot(N_e, others, '-',
         alpha=0.5, color='r',
         label='other')
plt.fill_between(N_e,
                 np.array(others)-np.array(others_std),
                 np.array(others)+np.array(others_std),
                 alpha=0.2, color='r')

plt.plot(N_e, gram, '-',
         alpha=0.5,
         label='grammatical')
plt.fill_between(N_e,
                 np.array(gram)-np.array(gram_std),
                 np.array(gram)+np.array(gram_std),
                 alpha=0.2)

plt.plot(N_e, sema, '-',
         alpha=0.5,
         label='semantic')
plt.fill_between(N_e,
                 np.array(sema)-np.array(sema_std),
                 np.array(sema)+np.array(sema_std),
                 alpha=0.2)

# 4. adjust figure parameters and save
fig_lettersize = 20
# plt.title('Plasticity effects')
plt.legend(loc='best', frameon=False, fontsize=fig_lettersize)
plt.xlabel(r'$N^{\rm E}$', fontsize=fig_lettersize)
plt.ylabel(r' incorrect sentences (%)', fontsize=fig_lettersize)
plt.xlim([100, 1500])
plt.ylim([0., 0.3])
plt.xticks(
    [200, 500, 800, 1100, 1400, 1700],
    ['$200$',  '$500$',  '$800$', '$1100$', '$1400$', '$1700$'],
    size=fig_lettersize,
    )
plt.yticks(
    [0., 0.1, 0.2, 0.3],
    ['$0$', '$10$', '$20$', '$30$'],
    size=fig_lettersize,
    )

if SAVE_PLOT:
    plots_dir = 'plots/GrammarTask/'
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
    plt.savefig(plots_dir+'error_types.pdf', format='pdf')
plt.show()
