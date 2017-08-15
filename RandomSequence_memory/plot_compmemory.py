import numpy as np
import matplotlib.pylab as plt
import cPickle as pickle
import os
import sys
from tempfile import TemporaryFile

import math
import sklearn
from sklearn import linear_model
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d

sys.path.insert(1, '.')
from utils import Bunch

def logistic_func(x, a, b, c, d):
    return a * np.tanh(b*x + c) + d

def fading_memory(perf_data):

    m = (perf_data[0]+perf_data[-1])/2
    xdata = perf_data
    ydata = np.arange(len(perf_data))
    f_interp = interp1d(xdata, ydata)
    return f_interp(m)

def log_func(x, a, b):
    return a*np.log(x)+ b

def fit_log(A, xdata, ydata):

    popt, pcov = curve_fit(log_func, xdata, ydata)
    print popt
    plt.plot(xdata, log_func(xdata, *popt), 'gray')


experiment_folder = 'RandomSequence_memory'
experiment_path = 'backup/' + experiment_folder + '/'
print '\nCalculating fading memory for the Sequence Learning Task...'

### figure parameters
width  =  7
height = 7
fig = plt.figure(1, figsize=(width, height))
fig_lettersize = 12

final_performance = []
final_A = []
final_N = []

L_values = np.array([50000])
N_values = np.array([100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100,
                     1200, 1400])
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
        final_N.append(int(N))

# t_past should be the same for every file
t_past = np.array(params.t_past)
final_performance = np.array(final_performance)
final_A = np.array(final_A)
final_N = np.array(final_N)

final_plot = np.zeros((len(np.unique(final_N)), len(np.unique(final_A))))
for i, n in enumerate(np.unique(final_N)):
    ind_n = np.where(final_N == n)[0]
    n_A = final_A[ind_n]
    n_performance = final_performance[ind_n]
    for j, a in enumerate(np.unique(n_A)):
        ind_a = np.where(n_A == a)[0]
        final_plot[i, j] = fading_memory(n_performance[ind_a].mean(0))

for i in xrange(final_plot.shape[1]):
    fit_log(np.unique(final_A)[i], N_values, final_plot[:, i])
    plt.errorbar(N_values, final_plot[:, i],
     fmt='-o', label=r'$A =$'+str(np.unique(final_A)[i]))

plt.title('SORN fading memory')
plt.xlabel(r'$N$', fontsize=fig_lettersize)
plt.ylabel('memory', fontsize=fig_lettersize)
plt.legend(loc='best')
# plt.ylim([0., 1.1])
# plt.xlim([0, t_past.max()+1])

# plt.savefig('plots/'+experiment_folder+'/memory_x_N.pdf', format='pdf')
plt.show()
