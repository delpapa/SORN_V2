""" Plot memory vs. network size"""

import os

import cPickle as pickle
import numpy as np
import matplotlib.pylab as plt
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d

# parameters to include in the plot
N_PAR = np.array([200])  # network sizes
A_PAR = np.array([4, 10, 50, 100, 200])               # input alphabet sizes
SAVE_PLOT = True

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
experiment_tag = '_perfxtplastc'
experiment_folder = 'RandomSequenceTask' + experiment_tag
experiment_path = 'backup/' + experiment_folder + '/'
experiment_n = len(os.listdir(experiment_path))

performance_list = np.zeros((experiment_n, 20))
t_list = np.zeros(experiment_n)
a_list = np.zeros(experiment_n)
for exp, exp_name in enumerate(os.listdir(experiment_path)):
    exp_n = [int(s) for s in exp_name.split('_') if s.isdigit()]
    stats = pickle.load(open(experiment_path+exp_name+'/stats.p', 'rb'))
    params = pickle.load(open(experiment_path+exp_name+'/init_params.p', 'rb'))
    performance_list[exp] = stats.performance
    a_list[exp] = int(params.A)
    t_list[exp] = int(params.steps_plastic)

# 2. filter plot data points
# filter A
if 'A_PAR' in globals():
    a_index = []
    for a in A_PAR:
        a_index.extend(np.where(a_list == a)[0].tolist())
    a_list = a_list[a_index]
    t_list = t_list[a_index]
    performance_list = performance_list[a_index]

# 3. plot memory vs. network size
memory = np.zeros( (len(np.unique(a_list)), len(np.unique(t_list))) )

# A loop
for j, a in enumerate(np.unique(a_list)):

    a_list_part = a_list[np.where(a_list == a)]
    t_list_part = t_list[np.where(a_list == a)]
    performance_list_part = performance_list[np.where(a_list == a)]
    for i, tplast in enumerate(np.unique(t_list_part)):
        perf_array = performance_list_part[np.where(t_list_part == tplast)].mean(0)
        memory[j, i] = fading_memory(perf_array)
    plt.plot(np.unique(t_list_part), memory[j], label=r'$A=%d$' %a)

# 4. adjust figure parameters and save
fig_lettersize = 12
plt.title('Random Sequence Task - Convergence')
plt.legend(loc='best')
plt.xlabel(r'$T_{\rm plast}$', fontsize=fig_lettersize)
plt.ylabel('Fading Memory', fontsize=fig_lettersize)
plt.xlim([0, 30000])
plt.ylim([0, 7])
plt.xticks(
    [0, 10000, 20000, 30000],
    ['0', '$1\cdot10^4$', '$2\cdot10^4$', '$3\cdot10^4$'],
    )

if SAVE_PLOT:
    plots_dir = 'plots/'+experiment_folder+'/'
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
        plt.savefig(plots_dir+'perfXtplast_A20.pdf', format='pdf')
plt.show()
