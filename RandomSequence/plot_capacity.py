import cPickle as pickle
import os
import sys
sys.path.insert(1, '.')
from tempfile import TemporaryFile

import numpy as np
import matplotlib.pylab as plt
import sklearn
from sklearn import linear_model
from scipy.optimize import curve_fit

from utils import Bunch

# parameters to include in the plot
# here, you should also select the x_min and x_max for each N, based on the
# plot_performance.py scritp. So far, they are determined by hand.
N_values = np.array([200])                       # network sizes
A_values = np.array([4])                         # input alphabet sizes
threshold = 0.98                                 # threshold for capacity
experiment_mark = ''                             # experiment mark


# Helper functions for the fits
def linear_fit(x, a, b):
    return a * x + b

def inv_linear_fit(y, a, b):
    return (y - b) / a

################################################################################
#                            Plot network capacity                             #
################################################################################

# 0. build figure
fig = plt.figure(1, figsize=(7, 7))

# 1. load performances and sequence lengths and calculate the fit for each one
print '\nCalculating capacity for the Sequence Learning Task...'
experiment_folder = 'RandomSequence' + experiment_mark
experiment_path = 'backup/' + experiment_folder + '/'
n_max = []
for n in N_values:

    partial_performance = []
    partial_L = []
    for experiment in os.listdir(experiment_path):

        # read data files
        N, L, A, _ = [s[1:] for s in experiment.split('_')]

        if n == int(N) and int(A) in A_values:
            exp_number = [int(s) for s in experiment.split('_') if s.isdigit()]
            params = pickle.load(open(experiment_path+experiment+'/sorn.p',
                                 'rb'))

            performance = params.RS_performance
            partial_performance.append(performance)
            partial_L.append(int(L))
    partial_L = np.array(partial_L)
    partial_performance = np.array(partial_performance)

    # linear fit
    # these values have been calculated based on the plot_performance script
    # if n == 100: x_min = 10; x_max = 50
    xdata = np.unique(partial_L)
    xdata = xdata[np.where(np.logical_and(xdata >= x_min , xdata <= x_max ))[0]]
    ydata = np.zeros(len(xdata))
    for i, l in enumerate(xdata):
        ind = np.where(partial_L == l)[0]
        ydata[i] = partial_performance[ind].mean()
    # fit a linear curve at the data, between the x_min and x_max limits
    popt, pcov = curve_fit(linear_fit, xdata, ydata)
    n_max.append(inv_linear_fit(threshold, *popt))

# 2. plot average performances and errors as a function of the sequence size
plt.plot(N_values, n_max, 'o')

# 3. edit figure properties
fig_lettersize = 12

plt.title('SORN capacitiy for the sequence learning task')
plt.xlabel('N', fontsize=fig_lettersize)
plt.ylabel(r'$n_{max}$', fontsize=fig_lettersize)

# 4. save figures
plt.savefig('plots/'+experiment_folder+'capacity_x_N.pdf', format='pdf')
plt.show()
