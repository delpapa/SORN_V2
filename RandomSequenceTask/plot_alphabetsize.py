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

def log_func(x, a, b):
    return a*np.log(x) + b

def fit_log(xdata, ydata):

    popt, pcov = curve_fit(log_func, xdata, ydata)
    log = interp1d(xdata, log_func(xdata, *popt))
    xnew = np.arange(50, 2000)
    plt.plot(xnew, log_func(xnew, *popt), color='gray', linestyle='-')

################################################################################
#                             Make plot                                        #
################################################################################

# calculate the fading memory with a threshold m
def fading_memory(perf_data):
    """Calculate how many past time steps are necessary for half performance"""

    m = 0.95 # memory threshold
    memory = np.zeros(len(perf_data))
    for i in xrange(len(perf_data)):
        memory[i] = np.where(perf_data[i] >= m)[0].size

    return memory.mean(), memory.std()

# 0. build figures
fig = plt.figure(1, figsize=(6, 6))

# 1. load performances and experiment parameters
print '\nCalculating memory for the Random Sequence Task...'
experiment_tag = '_memXsize_'
experiment_folder = 'RandomSequenceTask' + experiment_tag

a_list = []
n_list = []
performance = []

for j, i in enumerate(np.array([100, 200, 400, 800, 1600])):

    if i == 1600:
        experiment_path = 'backup/' + experiment_folder + 'N' + str(i) + '_moretraining/'
    else:
        experiment_path = 'backup/' + experiment_folder + 'N' + str(i) + '/'
    experiment_n = len(os.listdir(experiment_path))

    for exp, exp_name in enumerate(os.listdir(experiment_path)):
        exp_n = [int(s) for s in exp_name.split('_') if s.isdigit()]
        stats = pickle.load(open(experiment_path+exp_name+'/stats.p', 'rb'))
        params = pickle.load(open(experiment_path+exp_name+'/init_params.p', 'rb'))
        performance.append(stats.performance)
        a_list.append(stats.par.A)
        n_list.append(stats.par.N_e)

performance = np.array(performance)
a_list = np.array(a_list)
n_list = np.array(n_list)

for N in np.unique(n_list):
    a_list_red = a_list[n_list == N]
    n_list_red = n_list[n_list == N]
    performance_red = performance[n_list == N]

    memory_mean = np.zeros(np.unique(a_list_red).size)
    memory_std = np.zeros(np.unique(a_list_red).size)
    for i, A in enumerate(np.unique(a_list_red)):
        a_list_red_red = a_list_red[a_list_red == A]
        n_list_red_red = n_list_red[a_list_red == A]
        performance_red_red = performance_red[a_list_red == A]
        memory_mean[i], memory_std[i] = fading_memory(performance_red_red)

    if N in [100, 200, 400, 800, 1600]:
        print N
        print np.unique(a_list_red)
        print memory_mean
        if N == 100:
            #fit_log(np.unique(n_list_red), memory_mean)
            plt.plot(np.unique(a_list_red), memory_mean,
                marker='^',
                markersize='10',
                linestyle='None',
                color='k',
                label=str(N))
        if N == 200:
            #fit_log(np.unique(n_list_red), memory_mean)
            plt.plot(np.unique(a_list_red), memory_mean,
                marker='o',
                markersize='10',
                linestyle='None',
                color='k',
                label=str(N))
        if N == 400:
            #fit_log(np.unique(n_list_red), memory_mean)
            plt.plot(np.unique(a_list_red), memory_mean,
                marker='*',
                markersize='10',
                linestyle='None',
                color='k',
                label=str(N))
        if N == 800:
            #fit_log(np.unique(n_list_red), memory_mean)
            plt.plot(np.unique(a_list_red), memory_mean,
                marker='+',
                markersize='10',
                linestyle='None',
                color='k',
                label=str(N))
        if N == 1600:
            #fit_log(np.unique(n_list_red), memory_mean)
            plt.plot(np.unique(a_list_red), memory_mean,
                marker='.',
                markersize='10',
                linestyle='None',
                color='k',
                label=str(N))

# 4. adjust figure parameters and save
fig_lettersize = 15
# plt.title('Plasticity effects')
leg = plt.legend(loc='best', title='Network size', frameon=False, fontsize=fig_lettersize)
leg.get_title().set_fontsize(fig_lettersize)
plt.xlabel(r'$A$', fontsize=fig_lettersize)
plt.ylabel('$M_C$', fontsize=fig_lettersize)
#plt.xlim([90, 2000])
plt.xscale('log')
#plt.ylim([0, 8])
# plt.ylim([0, 4])
# plt.xticks(
#      [100, 1000],
#      ['$10^2$', '$10^3$'],
#      size=fig_lettersize,
#      )
#
# plt.yticks(
#     [0, 2, 4, 6, 8],
#     ['$0$', '$2$', '$4$', '$6$', '$8$'],
#     size=fig_lettersize,
#     )

if SAVE_PLOT:
    plots_dir = 'plots/'+experiment_folder+'/'
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
    plt.savefig(plots_dir+'memXnet2.pdf', format='pdf')
plt.show()
