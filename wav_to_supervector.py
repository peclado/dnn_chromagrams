#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    WAV to Supervector
    Luis Alvarado
    2017

    Convierte los archivos wav a los supervectores de entrada a la red.
"""
import numpy as np
import os
import Tools.funciones as tools

alpha_lnqt = 1.0
num_epochs = 10
training_dataset = 'Original'
# training_dataset = 'Anecoica'
batch_size = 512
eps = np.finfo(float).eps  # epsilon: Nº más pequeño (evitar divisiones por 0)
f_b0 = 30.87
f_inic = f_b0/(2**(1.0/24.0))  # B0 = 30.87 Hz, f_inic es el cuarto de tono anterior.
min_den_filt = 0.003
n_quarter_tones = 178
win_length = 4*2048
overlap = int(win_length*0.5)  # Traslape entre ventanas (50%)
n_fft = win_length
learning = np.zeros(num_epochs)
concatenated_frames = 15
l_supervector = concatenated_frames * n_quarter_tones

os.chdir("..")  # Regresar una carpeta
filez_train = np.genfromtxt('Database/train-set.txt', delimiter='; ', dtype='str', skip_header=1)
filez_test = np.genfromtxt('Database/test-set.txt', delimiter='; ', dtype='str', skip_header=1)
os.chdir("Database")
filez = np.concatenate((filez_test, filez_train), axis=0)
n_songs = np.size(filez, axis=0)
for idx in range(n_songs):

    filename = filez[idx, 0]
    outfile = filename[0:len(filename)-4] + "_" + str(alpha_lnqt)
    time_vector_outfile = filename[0:len(filename)-4] + "_timevector"
    in_sig, samp_freq, enc, n_samples = tools.read_audio(filename)
    #############################################################
    #                   Pre-processing                          #
    #############################################################
    print "Procesando archivo ", str(idx + 1), " de ", str(n_songs)
    if idx == 0:
        triang_filt_num, triang_filt_den = tools.triang_filters(f_inic, n_fft, n_quarter_tones, min_den_filt,
                                                                samp_freq)
    supervector, time_vector = tools.get_supervector_matrix(in_sig, samp_freq, n_samples, win_length, overlap, n_fft,
                                                            idx, f_inic, n_quarter_tones, min_den_filt, alpha_lnqt,
                                                            concatenated_frames, l_supervector, triang_filt_num,
                                                            triang_filt_den)
    # os.chdir("..")
    #np.savetxt(outfile + ".csv", supervector, delimiter=",", format=float32)
    np.save(outfile, supervector)
    np.save(time_vector_outfile, time_vector)

asdasaas = 34
