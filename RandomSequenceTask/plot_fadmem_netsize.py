"""
Plot fading memory as a function of network size
Curves for each alphabet size A
Network size variable
Read experiments from RST_memXsize_N*/ (N1600 -> _moretraining)
Save plots at RST_memXsize_/
This was used as Fig. 3A in the tentative ESANN abstract.
"""

import os
import sys
sys.path.insert(0, '')

import cPickle as pickle
import numpy as np
import matplotlib.pylab as plt
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d

# parameters to include in the plot
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
    print popt
    plt.plot(xnew, log_func(xnew, *popt), color='gray', linestyle='--')

from matplotlib import cm
start = 0.
stop = 1.
number_of_lines= 10
cm_subsection = np.linspace(start, stop, number_of_lines)

colors = [ cm.Blues(x) for x in cm_subsection ]

################################################################################
#                             Make plot                                        #
################################################################################

# calculate the fading memory with a threshold m
def fading_memory(perf_data):
    """Calculate how many past time steps are necessary for half performance"""

    m = 0.90 # memory threshold
    memory = np.zeros(len(perf_data))
    for i in xrange(len(perf_data)):
        memory[i] = np.where(perf_data[i] >= m)[0].size

    return memory.mean(), memory.std()

# 0. build figures
fig = plt.figure(1, figsize=(5, 5))

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

for A in np.unique(a_list):
    a_list_red = a_list[a_list == A]
    n_list_red = n_list[a_list == A]
    performance_red = performance[a_list == A]

    memory_mean = np.zeros(np.unique(n_list_red).size)
    memory_std = np.zeros(np.unique(n_list_red).size)
    for i, N in enumerate(np.unique(n_list_red)):
        a_list_red_red = a_list_red[n_list_red == N]
        n_list_red_red = n_list_red[n_list_red == N]
        performance_red_red = performance_red[n_list_red == N]
        memory_mean[i], memory_std[i] = fading_memory(performance_red_red)

    if A in [20, 30, 40]:
        print A
        #print np.unique(n_list_red)
        #print memory_mean
        if A == 20:
            fit_log(np.unique(n_list_red), memory_mean)
            plt.plot(np.unique(n_list_red), memory_mean,
                '-o',
                color=colors[5],
                markersize='10',
                label=str(A))
        if A == 30:
            fit_log(np.unique(n_list_red), memory_mean)
            plt.plot(np.unique(n_list_red), memory_mean,
                '-o',
                color=colors[6],
                markersize='10',
                label=str(A))
        if A == 40:
            fit_log(np.unique(n_list_red), memory_mean)
            plt.plot(np.unique(n_list_red), memory_mean,
                '-o',
                color=colors[7],
                markersize='10',
                label=str(A))

        # if A == 50:
        #     plt.plot(np.unique(n_list_red), memory_mean,
        #         marker='+',
        #         markersize='10',
        #         linestyle='None',
        #         color='k',
        #         label=str(A))
        #     fit_log(np.unique(n_list_red), memory_mean)
        # if A == 50:
        #     plt.plot(np.unique(n_list_red), memory_mean,
        #         marker='*',
        #         markersize='10',
        #         linestyle='None',
        #         color='k',
        #         label=str(A))
        #     # fit_log(np.unique(n_list_red), memory_mean)

# 4. adjust figure parameters and save
fig_lettersize = 15
# plt.title('Plasticity effects')
leg = plt.legend(loc='best', title='Alphabet size', frameon=False, fontsize=fig_lettersize)
leg.get_title().set_fontsize(fig_lettersize)
plt.xlabel(r'$\rm N^E$', fontsize=fig_lettersize)
plt.ylabel(r'$\rm M_{\tau}$', fontsize=fig_lettersize)
plt.xlim([90, 2000])
plt.xscale('log')
plt.ylim([0, 7])
# plt.ylim([0, 4])
plt.xticks(
     [100, 200, 400, 800, 1600],
     ['$100$', '$200$', '$400$', '$800$', '$1600$'],
     size=fig_lettersize,
     )

plt.yticks(
    [0, 2, 4, 6],
    ['$0$', '$2$', '$4$', '$6$'],
    size=fig_lettersize,
    )
plt.tight_layout()

if SAVE_PLOT:
    plots_dir = 'plots/ms/'
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
    plt.savefig(plots_dir+'fadmem_netsize.pdf', format='pdf')
plt.show()
