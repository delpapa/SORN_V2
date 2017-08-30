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
L_values = np.array([2000])                    # network sizes
A_values = np.array([4, 6, 10, 20, 40, 100])              # input alphabet sizes
experiment_tag = '_FadingMemory'# experiment tag

################################################################################
#                            Plot performance                                  #
################################################################################

# 0. build figures
fig = plt.figure(1, figsize=(7, 7))

# 1. load performances and sequence lengths
print '\nCalculating performance for the Sequence Learning Task...'
experiment_folder = 'RandomSequenceTask' + experiment_tag
experiment_path = 'backup/' + experiment_folder + '/'

all_performance = []
all_L = []
all_A = []
for experiment in os.listdir(experiment_path):

    # read data files and load performances
    N, L, A, _ = [s[1:] for s in experiment.split('_')]
    if int(L) in L_values and int(A) in A_values:
        print 'experiment', experiment, '...'

        exp_number = [int(s) for s in experiment.split('_') if s.isdigit()]
        perf = pickle.load(open(experiment_path+experiment+'/performance.p', 'rb'))

        # saves everything
        all_performance.append(perf)
        all_A.append(int(A))

all_A = np.array(all_A)
all_performance = np.array(all_performance)

# 2. plot average performances and errors as a function of the sequence size
for a in np.unique(all_A):
    ind_a = np.where(all_A == a)[0]

    red_performance = all_performance[ind_a].mean(0)

    plt.plot(red_performance, '-o', label = 'A ='+str(a))

# 3. edit figure properties
fig_lettersize = 12

# plt.axhline(y=0.98, color='gray', linestyle='--')
plt.title('Sequence Learning Task')
plt.legend(loc='best')
plt.xlabel(r'$t_{\rm past}$', fontsize=fig_lettersize)
plt.ylabel('Performance', fontsize=fig_lettersize)
plt.ylim([0., 1.1])

# 4. save figuresa
plots_dir = 'plots/'+experiment_folder+'/'
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)
# plt.savefig(plots_dir+'performance_x_L_N'+str(N_values[0])+'.pdf', format='pdf')
plt.show()
