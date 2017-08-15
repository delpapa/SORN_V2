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
N_values = np.array([200])                       # network sizes
A_values = np.arange(4, 27, 2)                  # input alphabet sizes
experiment_mark = '_R'                           # experiment mark

################################################################################
#                            Plot performance                                  #
################################################################################

# 0. build figures
fig = plt.figure(1, figsize=(7, 7))

# 1. load performances and sequence lengths
print '\nCalculating performance for the Sequence Learning Task...'
experiment_folder = 'RandomSequence' + experiment_mark
experiment_path = 'backup/' + experiment_folder + '/'

final_performance = []
final_L = []
final_A = []
for experiment in os.listdir(experiment_path):

    # read data files and load performances
    N, L, A, _ = [s[1:] for s in experiment.split('_')]
    if int(N) in N_values and int(A) in A_values:
        print 'experiment', experiment, '...'

        exp_number = [int(s) for s in experiment.split('_') if s.isdigit()]
        params = pickle.load(open(experiment_path+experiment+'/sorn.p', 'rb'))

        final_performance.append(params.RS_performance)
        final_L.append(int(L))
        final_A.append(int(A))
        # print 'Performance - LR:', '%.2f' % performance
final_L = np.array(final_L)
final_A = np.array(final_A)
final_performance = np.array(final_performance)

# 2. plot average performances and errors as a function of the sequence size
for a in np.unique(final_A):
    ind_a = np.where(final_A == a)[0]

    input_entropy = []
    performance = []
    performance_error = []

    for l in np.unique(final_L[ind_a]):
        ind_l = np.where(final_L[ind_a] == l)[0]

        input_entropy.append(l*np.log2(a))
        performance.append(final_performance[ind_a][ind_l].mean())
        # low_L = mean_L - np.percentile(final_performance[ind_a][ind_l], 25)
        # high_L = np.percentile(final_performance[ind_a][ind_l], 75) - mean_L
        performance_error.append(final_performance[ind_a][ind_l].std())

    input_entropy = np.array(input_entropy)
    performance = np.array(performance)
    plt.errorbar(input_entropy, performance, fmt='-o', label = 'A ='+str(a))

# 3. edit figure properties
fig_lettersize = 12

plt.axhline(y=0.98, color='gray', linestyle='--')
plt.title('Sequence Learning Task - Entropy')
plt.legend(loc='best')
plt.xlabel('Input entropy', fontsize=fig_lettersize)
plt.ylabel('Performance', fontsize=fig_lettersize)
plt.ylim([0.4, 1.1])

# 4. save figuresa
plots_dir = 'plots/'+experiment_folder+'/'
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)
plt.savefig(plots_dir+'performance_x_inputentropy_N'+str(N_values[0])+'.pdf',
            format='pdf')
plt.show()
