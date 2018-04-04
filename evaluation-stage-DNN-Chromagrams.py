#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    MAIN Chromagrams
    EVALUATION STAGE
    Luis Alvarado
    2017-2018
"""
import matplotlib.pyplot as plt
import mir_eval
import numpy as np
import os
import sys
import time
import Tools.funciones as tools
import Tools.chroma_vector as chroma_vec
import Tools.chroma_plots as chroma_plt

os.chdir("..")
num_epochs = sys.argv[1]
alpha_lnqt = sys.argv[2]
# num_epochs = 1
# alpha_lnqt = 0.0
# num_epochs = 100
dataset = 3
# alpha_lnqt = 0.0
if dataset == 1:
    res_dataset = 'Train'
    print 'Evaluación de set de Train.'
elif dataset == 2:
    res_dataset = 'Test'
    print 'Evaluación de set de Test.'
elif dataset == 3:
    res_dataset = 'Full'
    print 'Evaluación de base de datos completa.'

results_out = open('Resultados/Scores/Results_DNN_epochs_'+str(num_epochs)+'_alpha_'+str(alpha_lnqt)+'.out', 'w')
results_lst = ['Epochs', 'Alpha_LNQT', res_dataset, 'Condition', 'Root', 'MajMin', 'Thirds']
for item in results_lst:
    results_out.write("%s " % item)
results_out.write("\n")
# results_out.close()
res_per_song = ['Cancion', 'Epochs', 'Alpa LNQT', 'Root', 'MajMin', 'Thirds']
results_per_song = open('Resultados/Scores/Results_per_song_DNN_epochs_'+str(num_epochs)+'_alpha_'+str(alpha_lnqt)+'.out', 'w')
for item in res_per_song:
    results_per_song.write("%s " % item)
results_per_song.write("\n")

for jdx in range(5):
    tipo_reverb = jdx
    '''
    if tipo_reverb == 1:
        tipo_reverb = "Original"
    elif tipo_reverb == 2:
        tipo_reverb = "Anecoica"
    elif tipo_reverb == 3:
        tipo_reverb = "S1"
    elif tipo_reverb == 4:
        tipo_reverb = "S2"
    elif tipo_reverb == 5:
        tipo_reverb = "S3"
    elif tipo_reverb == 6:
        tipo_reverb = "S4"
    elif tipo_reverb == 7:
        tipo_reverb = "AulaMagna"
    elif tipo_reverb == 8:
        tipo_reverb = "Catedral"
    
    '''
    if tipo_reverb == 0:
        tipo_reverb = "Original"
    elif tipo_reverb == 1:
        # tipo_reverb = "Anecoica"
        tipo_reverb = "Anecoica_regrabacion_d32cm"
    elif tipo_reverb == 2:
        # tipo_reverb = "S1"
        tipo_reverb = "Anecoica_regrabacion_d64cm"
    elif tipo_reverb == 3:
        #  tipo_reverb = "S2"
        tipo_reverb = "Anecoica_regrabacion_d128cm"
    elif tipo_reverb == 4:
        # tipo_reverb = "S3"
        tipo_reverb = "Anecoica_regrabacion_d256cm"
    train_set = "train-set_conv_" + tipo_reverb + ".txt"
    test_set = "test-set_conv_" + tipo_reverb + ".txt"
    if tipo_reverb == "Original_Mono":
        tipo_reverb = ""
        train_set = "train-set.txt"
    elif tipo_reverb == "Original":
        test_set = "test-set.txt"
        train_set = "train-set.txt"
        conv_type = ""
    elif tipo_reverb == "Anecoica_regrabacion_d32cm":
        test_set = "test-set_anecoica_32.txt"
        train_set = "train-set_anecoica_32.txt"
        conv_type = "mono"
    elif tipo_reverb == "Anecoica_regrabacion_d64cm":
        test_set = "test-set_anecoica_64.txt"
        train_set = "train-set_anecoica_64.txt"
        conv_type = "mono"
    elif tipo_reverb == "Anecoica_regrabacion_d128cm":
        test_set = "test-set_anecoica_128.txt"
        train_set = "train-set_anecoica_128.txt"
        conv_type = "mono"
    elif tipo_reverb == "Anecoica_regrabacion_d256cm":
        test_set = "test-set_anecoica_256.txt"
        train_set = "train-set_anecoica_256.txt"
        conv_type = "mono"
    folder = tipo_reverb + "/"
    if dataset == 1:

        reference_lab = np.genfromtxt('Database/train-set.txt', delimiter='; ', dtype='str')
        estimated_lab = np.genfromtxt('Database/'+folder+train_set, delimiter='; ', dtype='str')
    elif dataset == 2:

        reference_lab = np.genfromtxt('Database/test-set.txt', delimiter='; ', dtype='str')
        estimated_lab = np.genfromtxt('Database/'+folder+test_set, delimiter='; ', dtype='str')
    elif dataset == 3:
        reference_lab1 = np.genfromtxt('Database/train-set.txt', delimiter='; ', dtype='str')
        reference_lab2 = np.genfromtxt('Database/test-set.txt', delimiter='; ', dtype='str', skip_header=1)
        estimated_lab1 = np.genfromtxt('Database/' + folder + train_set, delimiter='; ', dtype='str')
        estimated_lab2 = np.genfromtxt('Database/' + folder + test_set, delimiter='; ', dtype='str', skip_header=1)
        reference_lab = np.concatenate((reference_lab1, reference_lab2), axis=0)
        estimated_lab = np.concatenate((estimated_lab1, estimated_lab2), axis=0)
    arch_length = np.size(reference_lab, 0)
    eval_vec = np.zeros(arch_length)
    eval_vec_mean = np.zeros(arch_length)
    score_root_mean = 0.0
    score_thirds_mean = 0.0
    score_majmin_mean = 0.0
    for idx in range(arch_length - 1):
        ref_name = "Database/" + reference_lab[idx + 1, 1]
        est_name = estimated_lab[idx + 1, 1]
        out_name = est_name[6:-4]

        '''
        HAY QUE ARREGLAR ESTA PARTE DEL CÓDIGO. ADAPTARLA PARA LAS NUEVAS BASES DE DATOS.
        '''
        if 'Train' in estimated_lab[idx + 1, 1]:
            if tipo_reverb == "Original":
                est_name = "Resultados/" + folder + est_name[6:-4]+'__epoch_' + str(num_epochs) + \
                            '_alpha_' + str(alpha_lnqt) + "_out.lab"
            elif 'Anecoica_regrabacion' in tipo_reverb:
                est_name = "Resultados/" + folder + est_name[6:-4] + '_mono_epoch_' + str(num_epochs) + \
                           '_alpha_' + str(alpha_lnqt) + "_out.lab"
            else:
                est_name = "Resultados/" + folder + est_name[6:-4] + "_conv_" + tipo_reverb+'_epoch_'+str(num_epochs) + \
                            '_alpha_'+str(alpha_lnqt) + "_out.lab"
        elif 'Test' in estimated_lab[idx + 1, 1]:
            if tipo_reverb == "Original":
                est_name = "Resultados/" + folder + est_name[5:-4]+'__epoch_' + str(num_epochs) + \
                            '_alpha_' + str(alpha_lnqt) + "_out.lab"
            elif 'Anecoica_regrabacion' in tipo_reverb:
                est_name = "Resultados/" + folder + est_name[5:-4] + '_mono_epoch_' + str(num_epochs) + \
                           '_alpha_' + str(alpha_lnqt) + "_out.lab"
            else:
                est_name = "Resultados/" + folder + est_name[5:-4] + "_conv_" + tipo_reverb+'_epoch_'+str(num_epochs) + \
                            '_alpha_'+str(alpha_lnqt) + "_out.lab"
        ref_intervals, ref_labels = mir_eval.io.load_labeled_intervals(ref_name)
        est_intervals, est_labels = mir_eval.io.load_labeled_intervals(est_name)
        est_intervals, est_labels = mir_eval.util.adjust_intervals(est_intervals,
                                                                   est_labels,
                                                                   ref_intervals.min(),
                                                                   ref_intervals.max(),
                                                                   mir_eval.chord.NO_CHORD,
                                                                   mir_eval.chord.NO_CHORD)
        intervals, ref_labels, est_labels = mir_eval.util.merge_labeled_intervals(ref_intervals,
                                                                                  ref_labels,
                                                                                  est_intervals,
                                                                                  est_labels)
        durations = mir_eval.util.intervals_to_durations(intervals)
        comparisons_root = mir_eval.chord.root(ref_labels, est_labels)
        comparisons_majmin = mir_eval.chord.majmin(ref_labels, est_labels)
        comparisons_thirds = mir_eval.chord.thirds(ref_labels, est_labels)
        score_root = mir_eval.chord.weighted_accuracy(comparisons_root, durations)
        score_thirds = mir_eval.chord.weighted_accuracy(comparisons_thirds, durations)
        score_majmin = mir_eval.chord.weighted_accuracy(comparisons_majmin, durations)
        score_root_mean += score_root
        score_root_mean_for = score_root_mean / (idx + 1)
        score_thirds_mean += score_thirds
        score_majmin_mean += score_majmin
        eval_vec[idx] = score_root
        eval_vec_mean[idx] = score_root_mean_for
        res_per_song = [out_name, num_epochs, alpha_lnqt, score_root, score_majmin, score_thirds]
        print idx
        for item in res_per_song:
            results_per_song.write("%s " % item)
        results_per_song.write("\n")
        # print 'Score Roots: '+str(score_root)
        # print 'Score Thirds: '+str(score_thirds)
    score_thirds_mean = score_thirds_mean / (arch_length - 1)
    score_root_mean = score_root_mean / (arch_length - 1)
    score_majmin_mean = score_majmin_mean / (arch_length - 1)

    results_lst = [num_epochs, alpha_lnqt, res_dataset, tipo_reverb, score_root_mean, score_majmin_mean, score_thirds_mean]
    for item in results_lst:
        results_out.write("%s " % item)

    results_out.write("\n")

    print 'Score Roots Promedio '+tipo_reverb+': '+str(score_root_mean)
    print 'Score MajMin Promedio '+tipo_reverb+': '+str(score_majmin_mean)
    print 'Score Thirds Promedio '+tipo_reverb+': '+str(score_thirds_mean)
    plt.plot(eval_vec)
    plt.plot(eval_vec_mean)
    plt.show()
results_out.close()
results_per_song.close()
