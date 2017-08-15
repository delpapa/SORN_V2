import numpy as np
import matplotlib.pylab as plt
import cPickle as pickle
import os
import sys
sys.path.insert(1, '.')
from tempfile import TemporaryFile

import sklearn
from sklearn import linear_model
from scipy.optimize import curve_fit

from utils import Bunch

# parameters to include in the plot
N_values = np.array([200])                       # network sizes
L_values = np.array([50000])                     # input sequence sizes
experiment_mark = '_noPlasticity'                # experiment mark

################################################################################
#                            Plot network memory                               #
################################################################################

# 0. build figure
fig = plt.figure(1, figsize=(7, 7))

# 1. load performances
experiment_folder = 'RandomSequence_memory' + experiment_mark
experiment_path = 'backup/' + experiment_folder + '/'
print '\nCalculating memory for the Sequence Learning Task...'

final_performance = []
final_A = []
for experiment in os.listdir(experiment_path):

    # read data files
    N, L, A, _ = [s[1:] for s in experiment.split('_')]

    if int(L) in L_values and int(N) in N_values:
        print 'experiment ', experiment

        exp_number = [int(s) for s in experiment.split('_') if s.isdigit()]
        params = pickle.load(open(experiment_path+experiment+'/sorn.p', 'rb'))

        # performance is a list of numbers containing the performance
        # measured for t_past in [0, ..., 19]
        performance = params.RS_performance

        final_performance.append(performance)
        final_A.append(int(A))
t_past = np.array(params.t_past)                 # the same for every file
final_performance = np.array(final_performance)

# 2. plot average performances and errors as a function of t_past
for a in np.unique(final_A):
    ind = np.where(final_A == a)[0]
    mean_p = final_performance[ind].mean(0)
    std_p = final_performance[ind].std(0)
    low_p = mean_p - np.percentile(final_performance[ind], 5)
    high_p = np.percentile(final_performance[ind], 95) - mean_p
    # plt.plot(t_past, mean_p, '-o', label=r'$A=$'+str(a))
    plt.errorbar(t_past, mean_p, yerr=[std_p, std_p],
                 fmt='-o', label=r'$A=$'+str(a))

# 3. edit figure properties
fig_lettersize = 12

plt.title('SORN fading memory')
plt.xlabel(r'$t_{past}$', fontsize=fig_lettersize)
plt.ylabel('Performance', fontsize=fig_lettersize)
plt.legend(loc='best')
plt.ylim([0., 1.1])
plt.xlim([0, t_past.max()+1])

# 4. save figures
#plt.savefig('plots/'+experiment_folder+'/performance_x_tpast_N'\
#            +str(N_values[0])+'.pdf', format='pdf')
plt.show()
