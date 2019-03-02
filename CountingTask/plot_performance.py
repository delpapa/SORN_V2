import os
import pickle
import sys; sys.path.append('.')

import numpy as np
import matplotlib.pylab as plt

from utils import Bunch

# parameters to include in the plot
N_VALUES = np.array([200])              # network sizes
SAVE_PLOT = True

################################################################################
#                            Plot performance                                  #
################################################################################

# 0. build figure
fig = plt.figure(1, figsize=(7, 7))

# 1. load performances and sequence lengths
try:
    experiment_tag = sys.argv[1]
except:
    raise ValueError('Please specify a valid experiment tag.')

print('\nCalculating performance for the Counting Task...')
experiment_folder = 'CountingTask_{}'.format(experiment_tag)
experiment_path = 'backup/{}/'.format(experiment_folder)

final_performance = []
final_L = []
for experiment in os.listdir(experiment_path):

    # read data files and load performances
    N, L, _ = [s[1:] for s in experiment.split('_')]
    if int(N) in N_VALUES:
        print('experiment', experiment, '... ', end='')

        exp_number = [int(s) for s in experiment.split('_') if s.isdigit()]
        stats = pickle.load(open(experiment_path+experiment+'/stats.p', 'rb'))
        perf = stats.performance

        final_performance.append(perf)
        final_L.append(int(L))
        print('Performance - LR: {:.2f}'.format(perf))
final_L = np.array(final_L)
final_performance = np.array(final_performance)

# 2. plot average performances and errors as a function of the sequence size
for l in np.unique(final_L):
    ind = np.where(final_L == l)[0]
    mean_L = final_performance[ind].mean()
    low_L = mean_L - np.percentile(final_performance[ind], 25)
    high_L = np.percentile(final_performance[ind], 75) - mean_L
    std_L = final_performance[ind].std()
    plt.plot(l, mean_L, 'og')
    plt.errorbar(l, mean_L, yerr=std_L, color='g')

# 3. edit figure features
fig_lettersize = 15
plt.title('Counting Task', fontsize=fig_lettersize)
plt.xlabel('L', fontsize=fig_lettersize)
plt.ylabel('Performance', fontsize=fig_lettersize)
plt.xticks(fontsize=fig_lettersize)
plt.yticks(fontsize=fig_lettersize)
plt.ylim([0.4, 1.1])

# 4. save figures
if SAVE_PLOT:
    plots_dir = 'plots/{}/'.format(experiment_folder)
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
    plt.savefig('{}performance.pdf'.format(plots_dir), format='pdf')
plt.show()
