#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    MAIN Chromagrams
    TEST STAGE
    Luis Alvarado
    2017
"""
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import time
import Tools.funciones as tools
import Tools.chroma_vector as chroma_vec
import Tools.chroma_plots as chroma_plt

#####################################################
#            Import DNN-Parameters                  #
#####################################################
num_epochs = int(sys.argv[1])
# n_iterations = int(sys.argv[2])
alpha_lnqt = float(sys.argv[2])
# tipo_reverb = int(sys.argv[3])
# num_epochs = 2000
# alpha_lnqt = 0.0
tipo_reverb = 1
# dataset = int(sys.argv[2])
dataset = 3
if dataset == 1:
    print 'Test Stage: Set de Train.'
elif dataset == 2:
    print 'Test Stage: Set de Test.'
elif dataset == 3:
    print 'Test Stage: Base de datos completa.'
os.chdir("..")
b1 = np.loadtxt('Parametros-DNN/b1_'+'epoch_'+str(num_epochs)+'_alpha_'+str(alpha_lnqt) +
                '_.out', delimiter=',')
b2 = np.loadtxt('Parametros-DNN/b2_'+'epoch_'+str(num_epochs)+'_alpha_'+str(alpha_lnqt) +
                '_.out', delimiter=',')
b3 = np.loadtxt('Parametros-DNN/b3_'+'epoch_'+str(num_epochs)+'_alpha_'+str(alpha_lnqt) +
                '_.out', delimiter=',')
b4 = np.loadtxt('Parametros-DNN/b4_'+'epoch_'+str(num_epochs)+'_alpha_'+str(alpha_lnqt) +
                '_.out', delimiter=',')
W1 = np.loadtxt('Parametros-DNN/W1_'+'epoch_'+str(num_epochs)+'_alpha_'+str(alpha_lnqt) +
                '_.out', delimiter=',')
W2 = np.loadtxt('Parametros-DNN/W2_'+'epoch_'+str(num_epochs)+'_alpha_'+str(alpha_lnqt) +
                '_.out', delimiter=',')
W3 = np.loadtxt('Parametros-DNN/W3_'+'epoch_'+str(num_epochs)+'_alpha_'+str(alpha_lnqt) +
                '_.out', delimiter=',')
W4 = np.loadtxt('Parametros-DNN/W4_'+'epoch_'+str(num_epochs)+'_alpha_'+str(alpha_lnqt) +
                '_.out', delimiter=',')

#####################################################
#            Feature Extraction                     #
#####################################################

eps = np.finfo(float).eps  # epsilon: Nº más pequeño (evitar divisiones por 0)
f_b0 = 30.87
f_inic = f_b0/(2**(1.0/24.0))  # B0 = 30.87 Hz, f_inic es el cuarto de tono anterior.
min_den_filt = 0.003
n_quarter_tones = 178
win_length = 4*2048
overlap = int(win_length*0.5)  # Traslape entre ventanas (50%)
n_fft = win_length
concatenated_frames = 15
l_supervector = concatenated_frames * n_quarter_tones
l_chromagram = 12  # 12 notes in an octave

if tipo_reverb == 1:
    tipo_reverb = "Original"
elif tipo_reverb == 2:
    tipo_reverb = "Anecoica_regrabacion_d32cm"
elif tipo_reverb == 3:
    # tipo_reverb = "S1"
    tipo_reverb = "Anecoica_regrabacion_d64cm"
elif tipo_reverb == 4:
    #  tipo_reverb = "S2"
    tipo_reverb = "Anecoica_regrabacion_d128cm"
elif tipo_reverb == 5:
    # tipo_reverb = "S3"
    tipo_reverb = "Anecoica_regrabacion_d256cm"
elif tipo_reverb == 6:
    tipo_reverb = "S4"
elif tipo_reverb == 7:
    tipo_reverb = "AulaMagna"
elif tipo_reverb == 8:
    tipo_reverb = "Catedral"
print tipo_reverb
folder = tipo_reverb+"/"
test_set = "test-set_conv_"+tipo_reverb+".txt"
train_set = "train-set_conv_"+tipo_reverb + ".txt"


if tipo_reverb == "Original":
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
if dataset == 1:
    archivo = np.genfromtxt('Database/'+folder+train_set, delimiter='; ', dtype='str')
elif dataset == 2:
    archivo = np.genfromtxt('Database/'+folder+test_set, delimiter='; ', dtype='str')
elif dataset == 3:
    archivo1 = np.genfromtxt('Database/'+folder+train_set, delimiter='; ', dtype='str')
    archivo2 = np.genfromtxt('Database/'+folder+test_set, delimiter='; ', dtype='str', skip_header=1)
    archivo = np.concatenate((archivo1, archivo2), axis=0)
arch_length = np.size(archivo, 0)
arch = 0
total_accuracy = 0.0

for arch in range(arch_length - 1):
    os.chdir("Database")
    # print('Procesando archivo '+str(arch + 1)+' de '+str(arch_length - 1))
    sys.stdout.write('\r Procesando archivo: ' + str(arch + 1) + ' de ' + str(arch_length - 1) + '. alpha = '
                     + str(alpha_lnqt))
    arch_ = arch
    filename = folder+archivo[arch + 1, 0]
    g_tr_fname = folder+archivo[arch + 1, 1]
    g_tr_simple = archivo[arch + 1, 1]
    if 'Train' in g_tr_fname:
        out_txtname = folder+g_tr_simple[6:len(g_tr_simple)-4]+ '_' + conv_type+'_epoch_'+str(num_epochs) + \
                      '_alpha_'+str(alpha_lnqt)+'_out.lab'
    elif 'Test' in g_tr_fname:
        out_txtname = folder+g_tr_simple[5:len(g_tr_simple) - 4] + '_' + conv_type+'_epoch_'+str(num_epochs) + \
                      '_alpha_'+str(alpha_lnqt)+'_out.lab'
    '''
    in_sig, samp_freq, enc, n_samples = tools.read_audio(filename)
    #############################################################
    #                   Pre-processing                          #
    #############################################################
    """ TODO: Generar el supervector de STFT de 15 frames"""
    in_sig_power_spectra, time_vector = tools.get_power_spectra(in_sig, samp_freq, n_samples,
                                                                win_length, overlap, n_fft)
    n_frames = np.size(in_sig_power_spectra, 1)
    if arch_ == 0:
        triang_filt_num, triang_filt_den = tools.triang_filters(f_inic, n_fft, n_quarter_tones, min_den_filt, samp_freq)
    quarter_note_spectrogram_num = tools.get_quarter_note_spectrogram(in_sig_power_spectra, triang_filt_num,
                                                                      n_quarter_tones, n_frames, n_fft)
    toc = time.clock()
    quarter_note_spectrogram_den = tools.get_quarter_note_spectrogram(in_sig_power_spectra, triang_filt_den,
                                                                      n_quarter_tones, n_frames, n_fft)
    log_lnqt_spectrogram = np.divide(quarter_note_spectrogram_num, np.power(quarter_note_spectrogram_den + eps,
                                                                            alpha_lnqt))
    log_lnqt_spectrogram = np.log10(log_lnqt_spectrogram + 1)

    zeros_matrix = np.zeros(shape=(n_quarter_tones, (concatenated_frames - 1) / 2))
    matrix_for_concatenate = np.concatenate((zeros_matrix, log_lnqt_spectrogram, zeros_matrix), axis=1)
    supervector_matrix = np.zeros(shape=(l_supervector, n_frames))
    '''
    filename_npy = filename[0:len(filename) - 4] + "_" + str(alpha_lnqt) + ".npy"
    time_vector_npy = filename[0:len(filename) - 4] + "_timevector" + ".npy"
    # g_tr_fname = archivo[arch_, 1]
    supervector_matrix = np.load(filename_npy)
    time_vector = np.load(time_vector_npy)
    # sys.stdout.write('\r')
    # print " tiempo ejecución: ", tiempo_ejecucion
    n_frames = np.size(supervector_matrix, 1)
    ground_truth = tools.ground_truth_beatles(g_tr_fname, time_vector, n_frames)
    '''
    for idx in range(n_frames):
        aux_vec = matrix_for_concatenate[:, idx:idx + concatenated_frames]
        supervector_matrix[:, idx] = np.ndarray.flatten(aux_vec, 'F')
    '''
    x = supervector_matrix
    h1 = tools.relu_mat(np.matmul(W1, x) + b1[:, None])
    h2 = tools.relu_mat(np.matmul(W2, h1) + b2[:, None])
    h3 = tools.relu_mat(np.matmul(W3, h2) + b3[:, None])
    y = tools.sigmoid(np.matmul(W4, h3) + b4[:, None])
    # plt.imshow(y, origin='Origin', aspect="auto")
    # plt.show()
    ch_comparison = chroma_vec.chord_for_comparison()
    ch_comp_length = np.size(ch_comparison, 1)
    tic = time.clock()
    out_chord_idx = np.zeros(n_frames)
    out_chord_idx_alv = np.zeros(n_frames)
    distancia_alv = np.zeros(ch_comp_length)
    '''
    plt.figure(1)
    plt.imshow(y, origin='Origin', aspect='auto')
    plt.figure(2)
    plt.imshow(ground_truth, origin='Origin', aspect='auto')
    plt.show()
    '''
    for idx in range(n_frames):
        aux_y = y[:, idx]
        # distancia = np.linalg.norm(np.subtract(aux_y[:, None], ch_comparison), ord=2, axis=0)
        distancia = np.linalg.norm(aux_y[:, None] - ch_comparison, ord=2, axis=0)

        out_chord_idx[idx] = np.argmin(distancia)
        for jdx in range(ch_comp_length):
            asddasklj = ch_comparison[:, jdx]
            aux_dist_alv = distancia[jdx] / (np.sum(ch_comparison[:, jdx]) + 1)
            distancia_alv[jdx] = distancia[jdx] / (np.sum(ch_comparison[:, jdx]) + 1)
        out_chord_idx_alv[idx] = np.argmin(distancia_alv)
    '''
    Problema: LA DISTANCIA HACE QUE RECONOZCA COMO NO CHORDS A AQUELLOS ACORDES
    QUE SON RECONOCIDOS POR LA RED COMO ACORDES, SI SUS VALORES SON MENORES A 0.5.
    Solución: - Alternativa a distancia euclidiana d = d_euc / (sum(gt) + 1)
    - Distancia Canberra
    '''
    y_chord = chroma_vec.get_chord(out_chord_idx.tolist())
    y_chord_alv = chroma_vec.get_chord(out_chord_idx_alv.tolist())
    y_chord = y_chord_alv
    # acc = 0.0
    toc = time.clock()
    asasdas = toc - tic
    out_chord = np.array(range(n_frames), dtype='U25')
    out_net_txt = np.chararray(shape=(n_frames, 3), itemsize=12)
    out_net_txt[:, 0] = '0'
    previous_chord = 'X'
    contdx = 0
    out_eval = np.array([[]])
    for idx in range(n_frames):
        out_net_txt[idx, 2] = y_chord[idx]
        if previous_chord != out_net_txt[idx, 2] and idx == 0:
            previous_chord = out_net_txt[idx, 2]
        if previous_chord != out_net_txt[idx, 2] and idx > 0:
            # out_net_txt[idx + 1, 0] = time_vector[idx]
            if '0' not in out_eval:
                aux_time = out_net_txt[idx - 1, 1] = time_vector[idx - 1]
                out_eval = np.concatenate((out_eval, np.array([out_net_txt[idx - 1, :]])), axis=1)
                contdx = 0
            else:
                out_net_txt[idx - 1, 0] = time_vector[idx - 1 - contdx]
                aux_time = out_net_txt[idx - 1, 1] = time_vector[idx]
                out_eval = np.concatenate((out_eval, np.array([out_net_txt[idx - 1, :]])), axis=0)
                contdx = 0
            previous_chord = out_net_txt[idx, 2]
        else:
            contdx += 1
        if idx == n_frames - 1:
            if '0' not in out_eval:
                out_eval = np.array([[out_net_txt[0, 0], time_vector[idx], previous_chord]])
            else:
                out_net_txt[idx, 0] = aux_time
                out_net_txt[idx, 1] = time_vector[idx]
                out_net_txt[idx, 2] = previous_chord
                out_eval = np.concatenate((out_eval, np.array([out_net_txt[idx, :]])), axis=0)

    #if '0' not in out_eval:
    #    out_eval = np.array([[out_net_txt[0, 0], time_vector[idx], previous_chord]], dtype='U25')

    os.chdir("..")
    np.savetxt('Resultados/'+out_txtname, out_eval, fmt='%s', delimiter=' ')
    # total_accuracy += acc/n_frames
    # print('Acierto de ' + str(acc / n_frames*100) + '%')
# acc_eval = total_accuracy/arch_length*100
# print 'El acierto es de un '+str(acc_eval)+'% '
'''
plt.show()
plt.figure(1)
chroma_plt.chromagram_plot(ground_truth, time_vector)
plt.figure(2)
chroma_plt.chromagram_plot(y, time_vector)

plt.show()
'''
print('Done, my masta')
