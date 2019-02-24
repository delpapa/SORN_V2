"""Plot the events' duration and size distributions.

The plot work by thresholding the activity, given the parameter THETA (see
Del Papa et al. 2017 for details of the thresholding procedure)
"""

import pickle
import os
import sys; sys.path.append('.')
from tempfile import TemporaryFile

import numpy as np
import matplotlib.pylab as plt
import powerlaw as pl

# parameters to include in the plot
NUMBER_OF_FILES = 50

################################################################################
#                            Plot avalanches                                   #
################################################################################

def avalanche_distributions(activity, theta = 'half_mean'):
    """Calculate duration and size of all avalanches in activity.

    Returns two 1d-arrays containing the duration and size of each avalanche.

    Parameters:
        activity: 1d-array
            Array containing activity of a single experiment simulation
        theta: str
            Indicates which theta should be used to threshold the activity.
            Options are: 'half_mean': half of the mean activity (rounded down)
    """

    if theta == 'half_mean':
        theta = activity.mean()/2.

    thresholded_activity = activity - theta
    length_act = len(activity)
    duration_list = []
    size_list = []

    duration = 0
    size = 0
    for i in range(length_act):

        # add one time step to the current avalanche
        if thresholded_activity[i] > 0:
            duration += 1
            size += int(thresholded_activity[i])

        # finish current avalanche and prepare for the next one
        elif size != 0:
            duration_list.append(duration)
            size_list.append(size)
            duration = 0
            size = 0

    return np.asarray(duration_list), np.asarray(size_list)

if __name__ == "__main__":

    # 0. build figures
    fig = plt.figure(1, figsize=(12, 5))

    # 1. read files
    print('\nPlotting Neuronal Avalanche duration and size distributions...')
    experiment_folder = 'NeuronalAvalanches_{}'.format(sys.argv[1])
    experiment_path = 'backup/{}/'.format(experiment_folder)
    experiment = os.listdir(experiment_path)[0]
    N, sigma, _ = [s[1:] for s in experiment.split('_')]
    print('experiment', experiment, '...')

    # 2. load stats
    stats = pickle.load(open(experiment_path+experiment+'/stats.p', 'rb'), encoding='latin1')

    # 3. calculate neuronal avalanches
    T_data, S_data = avalanche_distributions(stats.activity)

    # 4. duration distribution
    fig_a = plt.subplot(121)
    T_x, T_inverse = np.unique(T_data, return_inverse=True)
    T_y_freq = plt.bincount(T_inverse)
    T_y = T_y_freq / float(T_y_freq.sum()) # normalization

    T_fit = pl.Fit(T_data, xmin=6, xmax=60, discrete=True)
    T_alpha = T_fit.alpha
    T_sigma = T_fit.sigma
    T_xmin = T_fit.xmin

    plt.plot(T_x, T_y, '.', markersize=2)
    pl.plot_pdf(T_data)
    T_fit.power_law.plot_pdf(label=r'$\alpha = %.2f$' %T_alpha)

    fig_lettersize = 12
    plt.title('Neuronal Avalanches - duration distribution')
    plt.legend(loc='best')
    plt.xlabel(r'$T$', fontsize=fig_lettersize)
    plt.ylabel(r'$f(T)$', fontsize=fig_lettersize)
    plt.xscale('log')
    plt.yscale('log')

    # 5. size distribution
    fig_b = plt.subplot(122)
    S_x, S_inverse = np.unique(S_data, return_inverse=True)
    S_y_freq = plt.bincount(S_inverse)
    S_y = S_y_freq / float(S_y_freq.sum()) # normalization

    S_fit = pl.Fit(S_data, xmin=10, xmax=1500, discrete=True)
    S_alpha = S_fit.alpha
    S_sigma = S_fit.sigma
    S_xmin = S_fit.xmin

    plt.plot(S_x, S_y, '.', markersize=2)
    pl.plot_pdf(S_data)
    S_fit.power_law.plot_pdf(label=r'$\tau = %.2f$' %S_alpha)

    fig_lettersize = 12
    plt.title('Neuronal Avalanches - size distribution')
    plt.legend(loc='best')
    plt.xlabel(r'$S$', fontsize=fig_lettersize)
    plt.ylabel(r'$f(S)$', fontsize=fig_lettersize)
    plt.xscale('log')
    plt.yscale('log')

    # 6. save figures
    plots_dir = 'plots/{}/'.format(experiment_folder)
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
        plt.savefig(plots_dir+'NAdistributions.pdf', format='pdf')

    plt.show()
