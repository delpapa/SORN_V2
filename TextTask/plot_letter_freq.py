import os
import sys; sys.path.append('.')

from collections import Counter
import pickle
import numpy as np
import matplotlib.pylab as plt

import common.stats


# parameters to include in the plot
N = np.array([200, 400, 1000, 10000], dtype=np.int)
SAVE_PLOT = True

################################################################################
#                             Make plot                                        #
################################################################################

# 0. build figures
fig = plt.figure(1, figsize=(10, 6))

with open(r'../data/M72_raw/corpus_simple.txt', "rt") as fin:
     corpus = fin.read().replace('\n', '').lower()
     chars_to_remove = '-&+1237=[]^_'
     for c in list(chars_to_remove):
         corpus = corpus.replace(c,'')

chars_input = Counter(corpus)
sorted_chars = [x[0] for x in sorted(chars_input.items(), key=lambda kv: kv[1], reverse=True)]
freq_input = np.array([chars_input[x] for x in sorted_chars])
freq_input = freq_input/freq_input.sum()

stats = {}
for n in N:
    experiment_tag = 'CHILDES_N{}_Tp500000_Tr30000'.format(n)
    experiment_folder = 'TextTask_' + experiment_tag
    experiment_path = 'backup/' + experiment_folder + '/'

    for exp, exp_name in enumerate(os.listdir(experiment_path)):
        exp_n = [int(s) for s in exp_name.split('_') if s.isdigit()]
        stats[str(exp)] = pickle.load(open(experiment_path+exp_name+'/stats.p', 'rb'))

    output = []
    for k, v in stats.items():
        output.append(v.output)
    output = ''.join(output)

    output_counter = Counter(output)
    freq_output = np.array([output_counter[x] for x in sorted_chars])
    freq_output = freq_output/freq_output.sum()

    plt.plot(np.arange(len(freq_input)), freq_output, linewidth='1', label='{}'.format(n))

plt.plot(np.arange(len(freq_input)), freq_input, linewidth='3', color='k', label='CHILDES')
leg = plt.legend(loc='best', frameon=False, fontsize=18)
leg.set_title('Network size',prop={'size':18})
plt.xticks(np.arange(len(sorted_chars)), sorted_chars, fontsize=15)
plt.yticks([0, 0.1, 0.2, 0.3],
           ['0%', '10%', '20%', '30%'], fontsize=18)
plt.ylim([0, 0.3])
plt.ylabel('frequency', fontsize=18)
plt.tight_layout()
if SAVE_PLOT:
    plots_dir = 'plots/TextTask/'
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
    plt.savefig(plots_dir+'CHILDES_letter_counts.pdf', format='pdf')
plt.show()
