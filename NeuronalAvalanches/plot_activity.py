"""Plot a sample of the sorn activity (last 1000 time steps)"""

import pickle
import os
import sys; sys.path.append('.')
from tempfile import TemporaryFile

import numpy as np
import matplotlib.pylab as plt

################################################################################
#                            Plot performance                                  #
################################################################################

# 0. build figures
fig = plt.figure(1, figsize=(6, 5))

# 1. read files
print('\nPlotting ConnectionFraction...')
experiment_folder = 'NeuronalAvalanches_{}'.format(sys.argv[1])
experiment_path = 'backup/{}/'.format(experiment_folder)
experiment = os.listdir(experiment_path)[0]
N, sigma, _ = [s[1:] for s in experiment.split('_')]
print('experiment', experiment, '...')

# 2. load stats
stats = pickle.load(open(experiment_path+experiment+'/stats.p', 'rb'))

# 3. plot connection fraction
plt.plot(stats.activity[-1000:], label='Activity')

# 3. edit figure properties
fig_lettersize = 12

plt.title('Neuronal Avalanches (spontaneous activity)')
plt.legend(loc='best')
plt.xlabel(r'time step', fontsize=fig_lettersize)
plt.ylabel(r'Activity', fontsize=fig_lettersize)

# 4. save figures
plots_dir = 'plots/{}/'.format(experiment_folder)
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)
plt.savefig(plots_dir+'activity.pdf', format='pdf')

plt.show()
