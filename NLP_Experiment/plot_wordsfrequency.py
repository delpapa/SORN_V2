"""
Plot relative number of errors and new sentences as a function of the SORN size
"""

import os
import sys
sys.path.insert(0, '')

from collections import Counter

import cPickle as pickle
import numpy as np
import matplotlib.pylab as plt

import common.stats


# parameters to include in the plot
SAVE_PLOT = True

################################################################################
#                             Make plot                                        #
################################################################################

# 0. build figures
fig = plt.figure(1, figsize=(7, 6))

# 1. load performances and experiment parameters
print '\nCalculating percent of wrong and new sentences... '

experiment_tag = '_test'
experiment_folder = 'NLP_Experiment' + experiment_tag
experiment_path = 'backup/' + experiment_folder + '/'
experiment_n = len(os.listdir(experiment_path))
outputs = []
net_size =  []

for exp, exp_name in enumerate(os.listdir(experiment_path)):

    exp_n = [int(s) for s in exp_name.split('_') if s.isdigit()]
    stats = pickle.load(open(experiment_path+exp_name+'/stats.p', 'rb'))
    params = pickle.load(open(experiment_path+exp_name+'/init_params.p', 'rb'))
    net_size.append(int(exp_name.split('_')[0][1:]))
    outputs.append(stats.output)

input_path = r'../data/M72_raw/corpus_simple.txt'
with open(input_path, "rt") as fin:
    input = fin.read().replace('\n', '')[:1000000]

words_input = Counter()
words_input.update(input.split())
input_mc = dict(words_input.most_common(10))
input_nwords = len(input.split())

words_output = Counter()
words_output.update(outputs[2].split())
output_mc = dict(words_output.most_common(10))
output_nwords = len(outputs[2].split())

import ipdb; ipdb.set_trace()

for t in input_mc:
   pass






new_color = 'blue'
wrong_color = 'red'

plt.plot(rem, new, '-o',
         color=new_color, alpha=0.5,
         label=r'$N^{\rm E}=400$ (new)')
plt.plot(rem, wrong, '-o',
         color=wrong_color, alpha=0.5,
         label=r'$N^{\rm E}=400$ (wrong)')

plt.fill_between(rem,
                 np.array(new)-np.array(new_std),
                 np.array(new)+np.array(new_std),
                 color=new_color, alpha=0.2)
plt.fill_between(rem,
                 np.array(wrong)-np.array(wrong_std),
                 np.array(wrong)+np.array(wrong_std),
                 color=wrong_color, alpha=0.2)

# 4. adjust figure parameters and save
fig_lettersize = 15
# plt.title('Plasticity effects')
plt.legend(loc='best', frameon=False, fontsize=fig_lettersize)
plt.xlabel(r'excluded sentences (%)', fontsize=fig_lettersize)
plt.ylabel(r'sentences (%)', fontsize=fig_lettersize)
plt.xlim([0, 64])
plt.ylim([0., 0.5])
plt.xticks(
    [16, 32, 48, 64],
    ['$25$', '$50$', '$75$', '$100$'],
    size=fig_lettersize,
)
plt.yticks(
    [0., 0.1, 0.2, 0.3, 0.4, 0.5],
    ['$0$', '$10$', '$20$', '$30$', '$40$', '$50$'],
    size=fig_lettersize,
)


if SAVE_PLOT:
    plots_dir = 'plots/' + experiment_folder + '/'
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
    plt.savefig(plots_dir+'removed_sentences.pdf', format='pdf')
plt.show()
