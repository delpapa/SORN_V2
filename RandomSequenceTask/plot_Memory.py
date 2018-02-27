""" Plot memory vs. network size"""

import os

import cPickle as pickle
import numpy as np
import matplotlib.pylab as plt
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from matplotlib import cm

# parameters to include in the plot
N_PAR = np.array([50, 100, 200, 400, 800, 1600, 2000])  # network sizes
A_PAR = np.array([4, 10, 20, 30, 40, 50])               # input alphabet sizes

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

################################################################################
#                             Make plot                                        #
################################################################################

def fading_memory(perf_data):
    """Calculate how many past time steps are necessary for half performance"""
    m = (perf_data[0]+perf_data[-1])/2
    xdata = perf_data
    ydata = np.arange(len(perf_data))
    f_interp = interp1d(xdata, ydata)
    return f_interp(m)

# 0. build figures
fig = plt.figure(1, figsize=(6, 5))

# 1. load performances and experiment parameters
print '\nCalculating memory for the Random Sequence Task...'
experiment_tag = '_FadingMemory'
experiment_folder = 'RandomSequenceTask' + experiment_tag
experiment_path = 'backup/' + experiment_folder + '/'
experiment_n = len(os.listdir(experiment_path))

performance_list = np.zeros((experiment_n, 20))
n_list = np.zeros(experiment_n)
a_list = np.zeros(experiment_n)
l_list = np.zeros(experiment_n)
for exp, exp_name in enumerate(os.listdir(experiment_path)):
    N, L, A, _ = [s[1:] for s in exp_name.split('_')]
    exp_n = [int(s) for s in exp_name.split('_') if s.isdigit()]
    perf = pickle.load(open(experiment_path+exp_name+'/performance.p', 'rb'))
    performance_list[exp] = perf
    n_list[exp] = int(N)
    a_list[exp] = int(A)
    l_list[exp] = int(L)

# 2. filter plot data points
# filter A
if 'A_PAR' in globals():
    a_index = []
    for a in A_PAR:
        a_index.extend(np.where(a_list == a)[0].tolist())
    a_list = a_list[a_index]
    n_list = n_list[a_index]
    performance_list = performance_list[a_index]

# filter N
if 'N_PAR' in globals():
    n_index = []
    for n in N_PAR:
        n_index.extend(np.where(n_list == n)[0].tolist())
    n_list = n_list[n_index]
    a_list = a_list[n_index]
    performance_list = performance_list[n_index]

# 3. plot memory vs. network size
for i, alphabet in enumerate(np.unique(a_list)):
    n_list_reduced = n_list[np.where(a_list == alphabet)]
    a_list_reduced = a_list[np.where(a_list == alphabet)]
    performance_list_reduced = performance_list[np.where(a_list == alphabet)]

    memory = np.zeros((len(np.unique(a_list)), len(np.unique(n_list_reduced))))
    for j, net_size in enumerate(np.unique(n_list_reduced)):
        perf_array = performance_list_reduced\
            [np.where(n_list_reduced == net_size)].mean(0)
        memory[i, j] = fading_memory(perf_array)

for i in range(len(np.unique(a_list))):
    plt.plot(memory[i])
import ipdb; ipdb.set_trace()

final_plot = np.zeros((len(np.unique(n_list)), len(np.unique(all_A))))
for i in xrange(final_plot.shape[1]):
    fit_log(np.unique(all_A)[i], N_values, final_plot[:, i])
    plt.errorbar(N_values, final_plot[:, i],
        fmt='o', label=r'$A =$'+str(np.unique(all_A)[i]))







for i, n in enumerate(np.unique(all_N)):
    ind_n = np.where(all_N == n)[0]
    n_A = all_A[ind_n]
    n_performance = all_performance[ind_n]
    for j, a in enumerate(np.unique(n_A)):
        ind_a = np.where(n_A == a)[0]

        final_plot[i, j] = fading_memory(n_performance[ind_a].mean(0))




# 4. adjust figure parameters and save
fig_lettersize = 12
plt.title('Random Sequence Task - Memory')
plt.legend(loc='best')
plt.xlabel(r'$N$', fontsize=fig_lettersize)
plt.ylabel('Mem', fontsize=fig_lettersize)

if SAVE_PLOT:
    plots_dir = 'plots/'+experiment_folder+'/'
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
        plt.savefig(plots_dir+'memory_vs_networksize.pdf', format='pdf')
plt.show()
