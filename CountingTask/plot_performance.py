import cPickle as pickle
import os
import sys
sys.path.insert(1, '.')
from tempfile import TemporaryFile

import numpy as np
import matplotlib.pylab as plt
import sklearn
from sklearn import linear_model

from utils import Bunch

# parameters to include in the plot
N_values = np.array([200])                       # network sizes
experiment_tag = '_SORN'                              # experiment tag

################################################################################
#                            Plot performance                                  #
################################################################################

# 0. build figure
fig = plt.figure(1, figsize=(7, 7))

# 1. load performances and sequence lengths
print '\nCalculating performance for the Counting Task...'
experiment_folder = 'CountingTask' + experiment_tag
experiment_path = 'backup/' + experiment_folder + '/'

final_performance = []
final_L = []
for experiment in os.listdir(experiment_path):

    # read data files and load performances
    N, L, _ = [s[1:] for s in experiment.split('_')]
    if int(N) in N_values:
        print 'experiment', experiment, '...',

        exp_number = [int(s) for s in experiment.split('_') if s.isdigit()]
        stats = pickle.load(open(experiment_path+experiment+'/stats.p', 'rb'))
        perf = stats.performance
        print perf
        # normalize performance as in Lazar et al. 2009, fig. 2
        max_perf = 1 - 0.5/(float(L)+2)
        final_performance.append(perf/max_perf)
        final_L.append(int(L))
        print 'Performance - LR:', '%.2f' % perf
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
fig_lettersize = 12

plt.title('Counting Task')
plt.xlabel('L', fontsize=fig_lettersize)
plt.ylabel('Performance', fontsize=fig_lettersize)
plt.ylim([0.4, 1.1])

# 4. save figures
plots_dir = 'plots/'+experiment_folder+'/'
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)
plt.savefig(plots_dir+'performance_x_L_N'+str(N_values[0])+'.pdf', format='pdf')
plt.show()
