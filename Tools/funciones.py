#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
FUNCIONES
Luis Alvarado (2017)
"""
from contextlib import closing
from multiprocessing import Pool
import multiprocessing as mp
import math
import matplotlib.pyplot as plt
import numpy as np
import pyfftw
import scikits.audiolab as au
import scipy as sp
# import tensorflow as tf
import chroma_vector as chroma_vec
import time
import threading


def get_numframes(n_samples, win_length, overlap):
    n_frames = int(math.floor((n_samples - overlap) / (win_length - overlap)))
    return n_frames


def frame_matrix(in_sig, win_length, overlap, n_frames):
    """ Regresa la matriz descompuesta en frames """
    """ Returns the frame-decomposed matrix"""
    # tic = time.time()
    in_sig_mat = np.zeros(shape=(win_length, n_frames))
    for idx in range(n_frames):
        in_sig_mat[:, idx] = in_sig[idx*overlap:idx*overlap + win_length]
    # toc = time.time()
    # print " framing: ", str(toc - tic)
    return in_sig_mat


def get_fft(x):
    num_fft = np.size(x, 0)
    return np.fft.fft(x, n=num_fft, axis=0)


def get_stft(in_sig, win_length, overlap, n_frames, n_fft):
    """ Regresa una matriz de STFT compleja """
    """ Returns a Complex STFT matrix"""
    # tic = time.time()
    in_sig_mat = frame_matrix(in_sig, win_length, overlap, n_frames)
    # toc = time.time()
    # print " frame matrix: ", str(toc - tic)
    # tic = time.time()
    hamm = np.hamming(win_length)
    # toc = time.time()
    # print " hamming wind: ", str(toc - tic)
    # tic = time.time()
    in_sig_mat = np.multiply(in_sig_mat, hamm[np.newaxis, :].T)  # hamm traspuesta
    # toc = time.time()
    # print " windowing: ", str(toc - tic)
    # tic = time.clock()
    # stft = np.fft.fft(in_sig_mat, n=n_fft, axis=0)
    # tic = time.time()
    stft = sp.fftpack.fft(in_sig_mat, n=n_fft, axis=0)
    # toc = time.time()
    # print " stft: ", str(toc - tic)
    return stft


def ground_truth_beatles(g_tr_fname, time_vector, n_frames):
    n_chords = 12
    g_tr_in = np.loadtxt(g_tr_fname, usecols=(0, 1))  # Ground-truth input matrix. Elimina texto indeseado.
    g_tr_chord = np.genfromtxt(g_tr_fname, dtype=None, comments=None)
    acorde = g_tr_chord[0]
    acorde = acorde[2]
    g_tr_mat = np.zeros(shape=(n_chords, n_frames))
    asd = g_tr_mat[:, 1]
    for jdx in range(n_frames):
        # print jdx
        for idx in range(np.size(g_tr_in, 0)):
            gr_t_inic = g_tr_in[idx, 0]
            gr_t_fin = g_tr_in[idx, 1]
            t = time_vector[jdx]
            ch = g_tr_chord[idx]

            # print ch
            if (gr_t_inic <= t) & (gr_t_fin >= t):
                ch = ch[2]
                ch = chroma_vec.flat_to_sharp(ch)
                aux_g_tr = chroma_vec.chroma_vector_beatles(ch)
                g_tr_mat[:, jdx] = chroma_vec.chroma_vector_beatles(ch)
                break
    return g_tr_mat


def mse(a, b):
    mse = np.divide(np.sum(np.power(np.abs(a-b), 2)), np.size(a))
    return mse


def read_audio(filename):
    in_sig, samp_freq, enc = au.wavread(filename)  # Lectura de audio
    # if np.size(in_sig, 1) == 2:
    #    in_sig = (in_sig[:, 0] + in_sig[:, 1]) / 2
    if enc == 'pcm16':
        nbits = 16
    if in_sig.ndim == 2:
        in_sig = (in_sig[:, 0] + in_sig[:, 1]) / 2
    n_samples = len(in_sig)
    return in_sig, samp_freq, nbits, n_samples


def triang_filters(f_inic, n_fft, n_quarter_tones, min_den_filt, samp_freq):
    """ Genera los filtros triangulares LNQT numerador y denominador a partir de la cantidad de bines de la FFT """
    """ Generates numerator/denominator LNQT filters by FFT bins"""
    #  LNQT: Locally Normalized Quarter-Tone
    # f_inic = f_inic/(2 ** (1 / 24.0))
    hz_vec_by_bin = np.linspace(float(samp_freq) / float(n_fft), samp_freq, n_fft)
    triang_filt_num = np.zeros(shape=(n_fft / 2, n_quarter_tones))
    triang_filt_den = np.zeros(shape=(n_fft / 2, n_quarter_tones))
    for jdx in range(n_quarter_tones):
        # f_inic = f_inic * (2 ** (1 / 24.0))
        f_c = f_inic * (2 ** (1.0 / 24.0))
        f_fin = f_inic * (2 ** (1.0 / 12.0))
        for idx in range(n_fft / 2):
            f_current = hz_vec_by_bin[idx]
            if (f_current >= f_inic) & (f_current <= f_c):
                triang_filt_num[idx, jdx] = (f_current - f_inic) / (f_c - f_inic)
                triang_filt_den[idx, jdx] = ((min_den_filt - 1.0) / (f_c - f_inic)) * (f_current - f_inic) + 1.0
            elif (f_current > f_c) & (f_current <= f_fin):
                triang_filt_num[idx, jdx] = (f_fin - f_current) / (f_fin - f_c)
                triang_filt_den[idx, jdx] = ((1.0 - min_den_filt) / (f_fin - f_c)) * (f_current - f_fin) + 1.0
            elif f_current > f_fin:
                break
        f_inic = f_inic * (2 ** (1 / 24.0))
    '''
    plt.figure(1)
    plt.plot(triang_filt_num)
    plt.figure(2)
    plt.plot(triang_filt_den)
    plt.figure(3)
    plt.plot(triang_filt_num)
    plt.plot(triang_filt_den)
    plt.show()
    '''
    return triang_filt_num, triang_filt_den


def get_power_spectra(in_sig, samp_freq, n_samples, win_length, overlap, n_fft):
    n_frames = get_numframes(n_samples, win_length, overlap)  # Nº de frames de la señal
    time_vector = np.linspace(float(win_length) / samp_freq, float(n_samples - overlap) / samp_freq, n_frames)
    # tic = time.time()
    in_sig_stft = get_stft(in_sig, win_length, overlap, n_frames, n_fft)
    # toc = time.time()
    # print " full stft: ", str(toc - tic)
    # in_sig_power_spectra = np.power(np.abs(in_sig_stft[0:n_fft / 2, :]), 2)
    # in_sig_stft2 = in_sig_stft[0:n_fft/2, :]
    # tic = time.time()
    # in_sig_power_spectra2 = np.abs(in_sig_stft[0:n_fft / 2, :])
    # toc = time.time()
    # print " power spectra 1", str(toc - tic)
    # tic = time.time()
    # in_sig_power_spectra = np.sqrt(in_sig_stft[0:n_fft/2, :].real**2 + in_sig_stft[0:n_fft/2, :].imag**2)
    # toc = time.time()
    # print "power spectra con raiz: ", str(toc - tic)
    # tic = time.time()
    in_sig_power_spectra = in_sig_stft[0:n_fft/2, :].real**2 + in_sig_stft[0:n_fft/2, :].imag**2  # |X|^2
    # toc = time.time()
    # print " power spectra sin raiz (al cuadrado)", str(toc - tic)
    # print " error power spectra: ", str(mse(in_sig_power_spectra, in_sig_power_spectra2))
    return in_sig_power_spectra, time_vector


def get_quarter_note_spectrogram(in_sig_power_spectra, triang_filt, n_quarter_tones, n_frames, n_fft):
    """ Genera los espectrogramas de cuarto de tono """
    """ Generates the Quarter-Tone Spectrograms"""
    # tic = time.time()
    quarter_note_spectrogram = np.sqrt(np.matmul(np.transpose(triang_filt), in_sig_power_spectra))

    # toc = time.time()
    # print " quarter-note alone: ", str(toc - tic)
    # tic = time.time()
    # quarter_note_spectrogram2 = (np.matmul(np.transpose(triang_filt), in_sig_power_spectra)) ** 0.5
    # toc = time.time()
    # print " quarter-note alone 2: ", str(toc - tic)
    # print (mse(quarter_note_spectrogram, quarter_note_spectrogram2))

    # error = mse(qt_spec_matrix, quarter_note_spectrogram)

    return quarter_note_spectrogram


def get_supervector_matrix(in_sig, samp_freq, n_samples, win_length, overlap, n_fft, arch_, f_inic, n_quarter_tones,
                           min_den_filt, alpha_lnqt, concatenated_frames, l_supervector, triang_filt_num,
                           triang_filt_den):
    eps = np.finfo(float).eps
    # tic_ejecucion = time.clock()
    in_sig_power_spectra, time_vector = get_power_spectra(in_sig, samp_freq, n_samples, win_length, overlap, n_fft)
    # toc_stft = time.clock()
    # tiempo_stft = toc_stft - tic_stft
    # print " full power spectra: ", str(tiempo_stft)
    n_frames = np.size(in_sig_power_spectra, 1)
    # tic = time.time()
    quarter_note_spectrogram_num = get_quarter_note_spectrogram(in_sig_power_spectra, triang_filt_num, n_quarter_tones,
                                                                n_frames, n_fft)
    quarter_note_spectrogram_den = get_quarter_note_spectrogram(in_sig_power_spectra, triang_filt_den, n_quarter_tones,
                                                                n_frames, n_fft)
    # toc = time.time()
    # print "getting quarter tone spec: ", str(toc - tic)
    # tic = time.time()
    # tic = time.time()
    # log_lnqt_spectrogram = np.divide(quarter_note_spectrogram_num, np.power(quarter_note_spectrogram_den + eps,
    #                                                                         alpha_lnqt))
    # toc = time.time()
    # print " log spec: ", str(toc - tic)
    # tic = time.time()
    log_lnqt_spectrogram = quarter_note_spectrogram_num / ((quarter_note_spectrogram_den + eps) ** alpha_lnqt)

    # toc = time.time()
    # print " log spec: ", str(toc - tic)
    # print mse(log_lnqt_spectrogram, log_lnqt_spectrogram2)
    log_lnqt_spectrogram = np.log10(log_lnqt_spectrogram + 1)
    # toc = time.time()
    # print " quarter-tone: ", str(toc - tic)
    # Supervectors matrix. Ok en optimización
    zeros_matrix = np.zeros(shape=(n_quarter_tones, (concatenated_frames - 1) / 2))  # ok
    matrix_for_concatenate = np.concatenate((zeros_matrix, log_lnqt_spectrogram, zeros_matrix), axis=1)  # ok
    supervector_matrix = np.zeros(shape=(l_supervector, n_frames))  # ok
    for idx in range(n_frames):  # ok
        aux_vec = matrix_for_concatenate[:, idx:idx + concatenated_frames]
        supervector_matrix[:, idx] = np.ndarray.flatten(aux_vec, 'F')
    return supervector_matrix, time_vector


def relu(x):
    l_x = len(x)
    for idx in range(l_x):
        if x[idx] <= 0:
            x[idx] = 0
    return x


def relu_mat(x):
    x[x < 0] = 0
    return x


def sigmoid(x):
    y = np.divide(1.0, 1 + np.exp(-x))
    return y

'''
def tf_log10(x):
    numerator = tf.log(x)
    denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator
'''
