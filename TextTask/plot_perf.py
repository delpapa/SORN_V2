import os
import sys; sys.path.append('.')

from collections import Counter
import pickle
import numpy as np
import matplotlib.pylab as plt

import common.stats


# parameters to include in the plot
N = np.array([100, 200], dtype=np.int)
chars_to_remove = "-&+1237=[]^_.!?,'"
SAVE_PLOT = True

################################################################################
#                             Make plot                                        #
################################################################################


fig = plt.figure(1, figsize=(7, 6))

with open(r'../data/M72_raw/corpus_simple.txt', "rt") as fin:
     corpus = fin.read().replace('\n', '').lower()
     chars_to_remove = '-&+1237=[]^_'
     for c in list(chars_to_remove):
         corpus = corpus.replace(c,'')

chars_input = Counter(corpus)
sorted_chars = [x[0] for x in sorted(chars_input.items(), key=lambda kv: kv[1], reverse=True)]
freq_input = np.array([chars_input[x] for x in sorted_chars])
freq_input = freq_input/freq_input.sum()
# remove spaces
sorted_chars = sorted_chars[1:]
freq_input = freq_input[1:]

stats = {}
for n in N:
    experiment_tag = 'CHILDES_N{}_Tp500000_Tr30000'.format(n)
    experiment_folder = 'TextTask_' + experiment_tag
    experiment_path = 'backup/' + experiment_folder + '/'

    stats[str(n)] = {}
    for exp, exp_name in enumerate(os.listdir(experiment_path)):
        exp_n = [int(s) for s in exp_name.split('_') if s.isdigit()]
        stats[str(n)][str(exp_n[0])] = pickle.load(open(experiment_path+exp_name+'/stats.p', 'rb'))

mean_perf = []
for i, (n, scores) in enumerate(stats.items()):

    n_elems = len(scores)
    new_scores = {}
    for j, value in scores['1'].spec_perf.items():
        new_scores[j] = 0
        for exp in range(n_elems):
            new_scores[j] += scores[str(exp+1)].spec_perf[j]/n_elems

    _ = new_scores.pop(' ')
    mean_perf.append(sum([dict(zip(sorted_chars, freq_input))[j]*new_scores[j] for j in new_scores]))

plt.plot(N, mean_perf, '-o', label='SORN')

stats = {}
for n in N:
    experiment_tag = 'CHILDES_N{}_RR_Tr30000'.format(n)
    experiment_folder = 'TextTask_' + experiment_tag
    experiment_path = 'backup/' + experiment_folder + '/'

    stats[str(n)] = {}
    for exp, exp_name in enumerate(os.listdir(experiment_path)):
        exp_n = [int(s) for s in exp_name.split('_') if s.isdigit()]
        stats[str(n)][str(exp_n[0])] = pickle.load(open(experiment_path+exp_name+'/stats.p', 'rb'))

mean_perf = []
for i, (n, scores) in enumerate(stats.items()):

    n_elems = len(scores)
    new_scores = {}
    for j, value in scores['1'].spec_perf.items():
        new_scores[j] = 0
        for exp in range(n_elems):
            new_scores[j] += scores[str(exp+1)].spec_perf[j]/n_elems

    _ = new_scores.pop(' ')
    mean_perf.append(sum([dict(zip(sorted_chars, freq_input))[j]*new_scores[j] for j in new_scores]))

plt.plot(N, mean_perf, '-o', label='Random network')


plt.legend(loc='best', frameon=False, fontsize=18)
plt.xticks(fontsize=18)
plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0],
           ['0%', '20%', '40%', '60%', '80%', '100%',], fontsize=18)
plt.ylim([0, 1])
plt.ylabel('avg. performance', fontsize=18)
plt.xlabel('Network size', fontsize=18)
plt.xscale('log')
plt.tight_layout()

if SAVE_PLOT:
    plots_dir = 'plots/TextTask/'
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
    plt.savefig(plots_dir+'CHILDES_perf.pdf', format='pdf')
plt.show()
