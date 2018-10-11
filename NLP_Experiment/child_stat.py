################################################################################
################################################################################

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import string
from collections import Counter

input_file = 'M72_raw/corpus_simple.txt'
input_len = 1000000
alphabet = 'abcdefghijklmnopqrstuvwxyz' # keep the alphabet limited
output_file_200 = 'SORN_output/CHILD_NE200.txt'
output_file_400 = 'SORN_output/CHILD_NE400.txt'
output_file_600 = 'SORN_output/CHILD_NE600.txt'

### preprocessing stuff
with open(input_file, "r") as fin:

    input_data = fin.read().replace('\n', '')[:input_len]
    letter_count_i = Counter(input_data)

with open(output_file_200, "r") as fin:
    out_data_200 = fin.read().replace('\n', '')
    letter_count_o200 = Counter(out_data_200)
with open(output_file_400, "r") as fin:
    out_data_400 = fin.read().replace('\n', '')
    letter_count_o400 = Counter(out_data_400)
with open(output_file_600, "r") as fin:
    out_data_600 = fin.read().replace('\n', '')
    letter_count_o600 = Counter(out_data_600)

### Count letters
t_count_i = 0; count_i = np.zeros(len(alphabet))
t_count_o200 = 0; count_o200 = np.zeros(len(alphabet))
t_count_o400 = 0; count_o400 = np.zeros(len(alphabet))
t_count_o600 = 0; count_o600 = np.zeros(len(alphabet))

for i, letter in enumerate(alphabet):

    t_count_i += letter_count_i[letter]
    t_count_o200 += letter_count_o200[letter]
    t_count_o400 += letter_count_o400[letter]
    t_count_o600 += letter_count_o600[letter]

    count_i[i] = letter_count_i[letter]
    count_o200[i] = letter_count_o200[letter]
    count_o400[i] = letter_count_o400[letter]
    count_o600[i] = letter_count_o600[letter]

ordered_count = np.argsort(count_i)[::-1]
ordered_alphabet = np.array([l for l in alphabet])[ordered_count]

### Count words
word_count_i = Counter(input_data.translate(None, string.punctuation).split())
word_count_o200 = Counter(out_data_200.translate(None, string.punctuation).split())
word_count_o400 = Counter(out_data_400.translate(None, string.punctuation).split())
word_count_o600 = Counter(out_data_600.translate(None, string.punctuation).split())
import ipdb; ipdb.set_trace()


plt.figure(1, figsize=(10, 5))
sub1 = plt.subplot(121)

plt.plot(count_o200[ordered_count] / t_count_o200 * 100, label = r'$N^E=200$')
plt.plot(count_o400[ordered_count] / t_count_o400 * 100, label = r'$N^E=400$')
plt.plot(count_o600[ordered_count] / t_count_o600 * 100, label = r'$N^E=600$')
plt.plot(count_i[ordered_count] / t_count_i * 100, 'k', linewidth=3, label='input')

fmt = '%.1f%%'
yticks = mtick.FormatStrFormatter(fmt)
sub1.yaxis.set_major_formatter(yticks)
plt.xticks(np.arange(0, 26, 1), ordered_alphabet)

plt.ylim([0, 18])
plt.legend(loc='best')
plt.title('Letter count')

plt.savefig('letter_stat.png', format='png')
plt.show()



# with open(input_file, "rt") as fin:
#     with open(output_file, "rt") as fout:
#
#         for line in fin:
#
#             import ipdb;ipdb.set_trace()
#
#             new_line = line.replace(' PERIOD xxx PERIOD', '.')
#             new_line = line.replace(' EXCLAIM xxx PERIOD', '!')
#             new_line = line.replace(' QUESTION xxx PERIOD', '?')
#             new_line = new_line.replace(' PERIOD','.')
#             new_line = new_line.replace(' COMMA',',')
#             new_line = new_line.replace(' QUESTION','?')
#             new_line = new_line.replace(' EXCLAIM','!')
#
#             new_line = new_line.replace(' xxx', '')
#             new_line = new_line.replace(' yyy', '')
#             new_line = new_line.replace(' www', '')
#             new_line = new_line.replace(' zzz', '')
#
#             new_line = new_line.replace(' PAUSE', '')
#
#             new_line = new_line.replace('MNAME','mname')
#             new_line = new_line.replace('FNAME','fname')
#             new_line = new_line.replace('ANAME','aname')
#
#             new_line = new_line.replace('. .','.')
#             new_line = new_line.replace('.?','?')
#             new_line = new_line.replace('?.','?')
#             new_line = new_line.replace('.!','!')
#             new_line = new_line.replace('!.','!')
#             new_line = new_line.replace(',.',',')
#             new_line = new_line.replace('.,',',')
#
#             fout.write(new_line)
