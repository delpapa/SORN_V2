import os
import sys
sys.path.insert(0, '')

from collections import Counter
import cPickle as pickle
import numpy as np
import matplotlib.pylab as plt

import common.stats


# parameters to include in the plot
SAVE_PLOT = False

################################################################################
#                             Make plot                                        #
################################################################################

# 0. build figures
fig = plt.figure(1, figsize=(7, 6))

# 1. load performances and experiment parameters
print '\n... '

experiment_tag = 'CHILDES'
experiment_folder = 'LanguageTask_' + experiment_tag
experiment_path = 'backup/' + experiment_folder + '/'
experiment_n = len(os.listdir(experiment_path))
net_size = np.zeros(experiment_n)
output = []

for exp, exp_name in enumerate(os.listdir(experiment_path)):
    exp_n = [int(s) for s in exp_name.split('_') if s.isdigit()]
    stats = pickle.load(open(experiment_path+exp_name+'/stats.p', 'rb'))
    net_size[exp] = int(exp_name.split('_')[0][1:])
    output.append(stats.output)
with open(r'../data/M72_raw/corpus_simple.txt', "rt") as fin:
     corpus = fin.read().replace('\n', '').lower()

words_input = Counter()
words_input.update(corpus.split())

import ipdb; ipdb.set_trace()

if SAVE_PLOT:
    plots_dir = 'plots/' + experiment_folder + '/'
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
    plt.savefig(plots_dir+'error_types.pdf', format='pdf')
plt.show()
