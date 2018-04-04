#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    MAIN Chromagrams
    Plot Results
    Luis Alvarado
    2017-2018
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import sys

# n_epochs = sys.argv[1]
# alpha_lnqt = sys.argv[2]
n_epochs = 60
alpha = [0.0, 0.25, 0.5, 0.75, 1.0]
os.chdir("..")
results_strings = np.array([[]])
results_array = np.array([[]])
for alpha_lnqt in alpha:
    if alpha_lnqt == 0.0:
        results_strings = np.genfromtxt('Resultados/Scores/Results_DNN_epochs_' + str(n_epochs) + '_alpha_' +
                                        str(alpha_lnqt)+'.out', delimiter=' ', dtype='str')

        results_array = np.genfromtxt('Resultados/Scores/Results_DNN_epochs_' + str(n_epochs) + '_alpha_' +
                                      str(alpha_lnqt)+'.out', delimiter=' ')
        reverb_types = np.size(results_array, 0) - 1
        #reverb_types = np.arange(aux)
    else:
        results_strings_for = np.genfromtxt('Resultados/Scores/Results_DNN_epochs_' + str(n_epochs) + '_alpha_' +
                                            str(alpha_lnqt) + '.out', delimiter=' ', dtype='str', skip_header=1)

        results_array_for = np.genfromtxt('Resultados/Scores/Results_DNN_epochs_' + str(n_epochs) + '_alpha_' +
                                          str(alpha_lnqt) + '.out', delimiter=' ', skip_header=1)

        results_array = np.concatenate((results_array, np.array(results_array_for)), axis=0)
        results_strings = np.concatenate((results_strings, results_strings_for), axis=0)
root = np.zeros(shape=(reverb_types, 5))
majmin = np.zeros(shape=(reverb_types, 5))
thirds = np.zeros(shape=(reverb_types, 5))
legend = np.chararray(reverb_types, itemsize=12)
for idx in range(reverb_types):
    legend[idx] = results_strings[idx + 1, 3]
for idx in range(len(alpha)):
    asd = results_array[1:9, 4]
    root[:, idx] = results_array[1+idx*reverb_types:9+idx*reverb_types, 4]
    majmin[:, idx] = results_array[1 + idx * reverb_types:9 + idx * reverb_types, 5]
    thirds[:, idx] = results_array[1 + idx * reverb_types:9 + idx * reverb_types, 6]
zssdas = 4
markers = ['.', 'x', '*', 9, 8, '>', '<', 'o']
plt.figure(1)
for idx in range(reverb_types):
    plt.plot(alpha, root[idx, :], marker=markers[idx], linewidth=1.0)
plt.xlabel(r'$\alpha_{LNQT}$')
plt.xticks(alpha)
plt.xlim([0, 1])
plt.ylabel("Score")
plt.ylim([0, 1])
plt.legend(legend)
plt.title('Score in Root Task for '+str(n_epochs) + ' epochs.')

plt.figure(2)
for idx in range(reverb_types):
    plt.plot(alpha, majmin[idx, :], marker=markers[idx], linewidth=1.0)
plt.xlabel(r'$\alpha_{LNQT}$')
plt.xticks(alpha)
plt.xlim([0, 1])
plt.ylabel("Score")
plt.ylim([0, 1])
plt.legend(legend)
plt.title('Score in Maj/Min Task for '+str(n_epochs) + ' epochs.')

plt.figure(3)
for idx in range(reverb_types):
    plt.plot(alpha, thirds[idx, :], marker=markers[idx], linewidth=1.0)
plt.xlabel(r'$\alpha_{LNQT}$')
plt.xticks(alpha)
plt.xlim([0, 1])
plt.ylabel("Score")
plt.ylim([0, 1])
plt.legend(legend)
plt.title('Score in Thirds Task for '+str(n_epochs) + ' epochs.')
plt.show()