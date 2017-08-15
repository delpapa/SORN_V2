import cPickle as pickle
import os
import sys
sys.path.insert(1, '.')
from tempfile import TemporaryFile

import numpy as np
import matplotlib.pylab as plt
import sklearn
from sklearn import linear_model

# parameters to include in the plot
N_values = np.array([50])                       # network sizes
A_values = np.arange(2, 52, 1)                  # input alphabet sizes
experiment_mark = '_capacity'                           # experiment mark

################################################################################
#                            Plot performance                                  #
################################################################################

# 0. build figures
fig = plt.figure(1, figsize=(7, 7))

# 1. load performances and sequence lengths
print '\nCalculating performance for the Sequence Learning Task...'
experiment_folder = 'RandomSequence' + experiment_mark
experiment_path = 'backup/' + experiment_folder + '/'

all_performance = []
all_L = []
all_A = []
for experiment in os.listdir(experiment_path):

    # read data files and load performances
    N, L, A, _ = [s[1:] for s in experiment.split('_')]
    if int(N) in N_values and int(A) in A_values:
        print 'experiment', experiment, '...'

        exp_number = [int(s) for s in experiment.split('_') if s.isdigit()]
        params = pickle.load(open(experiment_path+experiment+'/sorn.p', 'rb'))

        all_performance.append(params.RS_performance)
        all_L.append(int(L))
        all_A.append(int(A))
        # print 'Performance - LR:', '%.2f' % performance
all_L = np.array(all_L)
all_A = np.array(all_A)
all_performance = np.array(all_performance)

# 2. plot average performances and errors as a function of the sequence size

import ipdb; ipdb.set_trace()
for a in np.unique(all_A):
    ind_a = np.where(all_A == a)[0]

    L = []
    performance = []
    log_error = []
    performance_error = []

    for l in np.unique(all_L[ind_a]):
        ind_l = np.where(all_L[ind_a] == l)[0]

        L.append(l)
        performance.append(all_performance[ind_a][ind_l].mean())
        log_error.append(1. - performance[-1])
        performance_error.append(all_performance[ind_a][ind_l].std())

    plt.errorbar(L, performance, fmt='o', label = 'A ='+str(a))

# 3. edit figure properties
fig_lettersize = 12

# plt.axhline(y=0.98, color='gray', linestyle='--')
plt.title('Sequence Learning Task')
plt.legend(loc='best')
plt.xlabel('L', fontsize=fig_lettersize)
plt.ylabel('Performance', fontsize=fig_lettersize)
plt.ylim([0.4, 1.1])

# 4. save figuresa
plots_dir = 'plots/'+experiment_folder+'/'
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)
plt.savefig(plots_dir+'performance_x_L_N'+str(N_values[0])+'.pdf', format='pdf')
plt.show()
