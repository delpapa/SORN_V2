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
NU_values = np.arange(0, 201, 1)               # input pool sizes
A_values = np.arange(4, 53, 1)                   # input alphabet sizes
experiment_mark = '_NU'                    # experiment mark

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
all_NU = []
all_A = []
for experiment in os.listdir(experiment_path):

    # read data files and load performances
    N, NU, A, _ = [s[1:] for s in experiment.split('_')]
    NU = NU[1:]  # remove 'U'
    if int(NU) in NU_values and int(A) in A_values:
        print 'experiment', experiment, '...'

        exp_number = [int(s) for s in experiment.split('_') if s.isdigit()]
        params = pickle.load(open(experiment_path+experiment+'/sorn.p', 'rb'))

        all_performance.append(params.RS_performance)
        all_NU.append(int(NU))
        all_A.append(int(A))
        # print 'Performance - LR:', '%.2f' % performance
all_NU = np.array(all_NU)
all_A = np.array(all_A)
all_performance = np.array(all_performance)

# 2. plot average performances and errors as a function of the sequence size
for a in np.unique(all_A):
    ind_a = np.where(all_A == a)[0]

    NU = []
    performance = []
    performance_error = []

    for nu in np.unique(all_NU[ind_a]):
        ind_nu = np.where(all_NU[ind_a] == nu)[0]

        NU.append(nu)
        performance.append(all_performance[ind_a][ind_nu].mean())
        performance_error.append(all_performance[ind_a][ind_nu].std())

    plt.errorbar(NU, performance, fmt='-', label = r'$A =$'+str(a))

# 3. edit figure properties
fig_lettersize = 12

plt.title('Sequence Learning Task')
plt.legend(loc='best')
plt.xlabel(r'$N^U$', fontsize=fig_lettersize)
plt.ylabel('Performance', fontsize=fig_lettersize)
plt.ylim([0., 1.1])
# plt.yscale('log')

# 4. save figuresa
plots_dir = 'plots/'+experiment_folder+'/'
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)
plt.savefig(plots_dir+'performance_x_NU_N'+str(N_values[0])+'.pdf', format='pdf')
plt.show()
