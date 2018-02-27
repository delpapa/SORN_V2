"""
Plot error as a function of sequence size L
Curves for each network size N
Alphabet size fixed in A=20
Read experiments from RST_memXL_N*/
Save plots at memXL/
This was used as Fig. 2A in the tentative ESANN abstract.
"""

import os

import cPickle as pickle
import numpy as np
import matplotlib.pylab as plt
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d

# parameters to include in the plot
SAVE_PLOT = True

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
NETSIZE = ['100', '200', '400', '800']
max_L = []
max_netsize = np.array([100, 200, 400, 800])
for netsize in NETSIZE:
    experiment_tag = '_memXL_N' + netsize
    experiment_folder = 'RandomSequenceTask' + experiment_tag 
    experiment_path = 'backup/' + experiment_folder + '/'
    experiment_n = len(os.listdir(experiment_path))


    performance_list = np.zeros((experiment_n, 100))
    l_list = np.zeros(experiment_n)
    for exp, exp_name in enumerate(os.listdir(experiment_path)):

        exp_n = [int(s) for s in exp_name.split('_') if s.isdigit()]
        stats = pickle.load(open(experiment_path+exp_name+'/stats.p', 'rb'))
        params = pickle.load(open(experiment_path+exp_name+'/init_params.p', 'rb'))

        performance_list[exp] = stats.performance
        l_list[exp] = int(params.L)

        # 3. plot memory vs. network size
        memory_mean = np.zeros( len(np.unique(l_list)) )
        memory_std = np.zeros( len(np.unique(l_list)) )
        perf_array_last = np.zeros( len(np.unique(l_list)) )

    for i, l in enumerate(np.unique(l_list)):

        # if not (netsize == '800' and l == 1450)
        perf_array = performance_list[np.where(l_list == l)]
        memory_mean[i], memory_std[i] = fading_memory(perf_array)
        perf_array_last[i] = perf_array[-1].mean()

    information = np.unique(l_list)

    if netsize == '100':
        plt.plot(information, 1-perf_array_last, '--k', label=netsize)
    elif netsize == '200':
        plt.plot(information, 1-perf_array_last, '-k', label=netsize)
    elif netsize == '400':
        plt.plot(information, 1-perf_array_last, '-.k', label=netsize)
    if netsize == '800':
        perf_array_last[np.where(information == 1450)] = 1
        plt.plot(information, 1-perf_array_last, ':k', label=netsize)

    max_L.append(information[np.where(perf_array_last < 0.9)[0][0]])

# plt.plot(max_netsize, max_L)


# 4. adjust figure parameters and save
fig_lettersize = 15
# plt.title('Random Sequence Task - Sequence Size')
leg = plt.legend(loc='best', title='Network size', frameon=False, fontsize=fig_lettersize)
leg.get_title().set_fontsize(fig_lettersize)
plt.xlabel(r'$L$', fontsize=fig_lettersize)
plt.ylabel('Error', fontsize=fig_lettersize)
plt.axhline(y=.1, xmin=0, xmax=5000, linewidth=1, color = 'gray', alpha=0.5)
plt.xlim([0, 3000])
plt.ylim([0, 1])
plt.xticks(
     [0, 1000, 2000, 3000],
     ['$0$', '$1000$', '$2000$', '$3000$'],
     fontsize=fig_lettersize
     )
plt.yticks(
    [0, 0.2, 0.4, 0.6, 0.8],
    ['$0$', '$0.2$', '$0.4$', '$0.6$', '$0.8$'],
    fontsize=fig_lettersize,
    )

if SAVE_PLOT:
    plots_dir = 'plots/RandomSequenceTask_memXL/'
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
    plt.savefig(plots_dir+'memXinf.pdf', format='pdf')
plt.show()
