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
from scipy.interpolate import interp1d

# parameters to include in the plot
N_values = np.array([50, 100, 200, 400, 800, 1600, 2000])         # network sizes
A_values = np.array([4, 10, 20, 30, 40, 50]) # input alphabet sizes
L_values = np.array([50000])                          # sequence sizes
experiment_tag = '_FadingMemory'                      # experiment tag

################################################################################
#                            Plot performance                                  #
################################################################################

def log_interp1d(xx, yy, kind='linear'):
    logx = np.log10(xx)
    logy = np.log10(yy)
    lin_interp = sp.interpolate.interp1d(logx, logy, kind=kind)
    log_interp = lambda zz: np.power(10.0, lin_interp(np.log10(zz)))
    return log_interp

def fading_memory(perf_data):

    m = (perf_data[0]+perf_data[-1])/2
    xdata = perf_data
    ydata = np.arange(len(perf_data))
    f_interp = interp1d(xdata, ydata)
    return f_interp(m)

def log_func(x, a, b):
    return a*np.log(x) + b

def fit_log(A, xdata, ydata):

    popt, pcov = curve_fit(log_func, xdata, ydata)
    log = interp1d(xdata, log_func(xdata, *popt))
    xnew = np.arange(50, 2000)
    plt.plot(xnew, log_func(xnew, *popt), 'k--')

# 0. build figures
fig = plt.figure(1, figsize=(5, 5))

# 1. load performances and sequence lengths
print '\nCalculating performance for the Sequence Learning Task...'
experiment_folder = 'RandomSequenceTask' + experiment_tag
experiment_path = 'backup/' + experiment_folder + '/'

all_performance = []
all_N = []
all_A = []
all_L = []
for experiment in os.listdir(experiment_path):

    # read data files and load performances
    N, L, A, _ = [s[1:] for s in experiment.split('_')]
    if int(N) in N_values and int(A) in A_values and int(N) in N_values:
        print 'experiment', experiment, '...'

        exp_number = [int(s) for s in experiment.split('_') if s.isdigit()]
        perf = pickle.load(open(experiment_path+experiment+'/performance.p', 'rb'))

        # saves everything
        all_performance.append(perf)
        all_A.append(int(A))
        all_N.append(int(N))
        all_L.append(int(L))

all_A = np.array(all_A)
all_N = np.array(all_N)
all_L = np.array(all_L)
all_performance = np.array(all_performance)

final_plot = np.zeros((len(np.unique(all_N)), len(np.unique(all_A))))
for i, n in enumerate(np.unique(all_N)):
    ind_n = np.where(all_N == n)[0]
    n_A = all_A[ind_n]
    n_performance = all_performance[ind_n]
    for j, a in enumerate(np.unique(n_A)):
        ind_a = np.where(n_A == a)[0]

        final_plot[i, j] = fading_memory(n_performance[ind_a].mean(0))

for i in xrange(final_plot.shape[1]):
    fit_log(np.unique(all_A)[i], N_values, final_plot[:, i])
    plt.errorbar(N_values, final_plot[:, i],
     fmt='o', label=r'$A =$'+str(np.unique(all_A)[i]))


# 3. edit figure properties
fig_lettersize = 12

plt.title('Sequence Learning Task')
plt.legend(loc='best')
plt.xlabel(r'$N$', fontsize=fig_lettersize)
plt.ylabel('M', fontsize=fig_lettersize)
plt.xlim(0, 2000)

# 4. save figuresa
plots_dir = 'plots/'+experiment_folder+'/'
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)
plt.savefig(plots_dir+'memory_x_N.pdf', format='pdf')
plt.show()
