#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    MAIN Chromagrams
    Luis Alvarado
    2017
"""

import Tools.funciones as tools
import Tools.chroma_vector as chroma_vec
import Tools.chroma_plots as chroma_plt
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy as sp
import scipy.signal as sg
import scikits.audiolab as au
import sys
import tensorflow as tf
import time
from io import StringIO
plt.rcParams['agg.path.chunksize'] = 10000  # Poder plotear grandes vectores

#############################################################
#                  Definiciones previas                     #
#############################################################


num_epochs = int(sys.argv[1])
alpha_lnqt = float(sys.argv[2])
# n_iterations = int(sys.argv[2])
# alpha_lnqt = 0.0
# num_epochs = 1
# n_iterations = 1
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
#############################################################
#                  DNN-parameters definition                #
#############################################################
concatenated_frames = 15
l_supervector = concatenated_frames * n_quarter_tones
l_chromagram = 12  # 12 notes in an octave
l_hidden = 512
learning_rate = 0.1
keep_prob = 0.5
dim = None
sess = tf.InteractiveSession()
# Input
in_layer = tf.placeholder(tf.float32, shape=[l_supervector, dim])
# Output Ground-Truth
out_gt = tf.placeholder(tf.float32, shape=[l_chromagram, dim])
# DNN Parameters
W1 = tf.Variable(tf.truncated_normal([l_hidden, l_supervector], mean=0.0, stddev=0.1))
W2 = tf.Variable(tf.truncated_normal([l_hidden, l_hidden], mean=0.0, stddev=0.1))
W3 = tf.Variable(tf.truncated_normal([l_hidden, l_hidden], mean=0.0, stddev=0.1))
W4 = tf.Variable(tf.truncated_normal([l_chromagram, l_hidden], mean=0.0, stddev=0.1))
b1 = tf.Variable(tf.zeros(shape=(l_hidden, 1)))
b2 = tf.Variable(tf.zeros(shape=(l_hidden, 1)))
b3 = tf.Variable(tf.zeros(shape=(l_hidden, 1)))
b4 = tf.Variable(tf.zeros(shape=(l_chromagram, 1)))
# hidden layers
hid_lay1 = tf.nn.relu(tf.add(tf.matmul(W1, in_layer), b1))
hid_lay1 = tf.nn.dropout(hid_lay1, keep_prob=keep_prob)
hid_lay2 = tf.nn.relu(tf.add(tf.matmul(W2, hid_lay1), b2))
hid_lay2 = tf.nn.dropout(hid_lay2, keep_prob=keep_prob)
hid_lay3 = tf.nn.relu(tf.add(tf.matmul(W3, hid_lay2), b3))
hid_lay3 = tf.nn.dropout(hid_lay3, keep_prob=keep_prob)
# output layer
out_layer = tf.nn.sigmoid(tf.matmul(W4, hid_lay3) + b4)
# RUN DNN
cross_entropy = tf.divide(-tf.reduce_sum(tf.add(tf.multiply(out_gt, tf.log(out_layer+eps)),
                                                tf.multiply(1-out_gt+eps, tf.log(1-out_layer+eps)))),
                          tf.constant(value=12.0 * np.size(out_gt)))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
tf.global_variables_initializer().run()

#############################################################
#                   Input                                   #
#############################################################

os.chdir("..")  # Regresar una carpeta
if training_dataset == 'Original':
    archivo = np.genfromtxt('Database/train-set.txt', delimiter='; ', dtype='str', skip_header=1)
    os.chdir("Database")
if training_dataset == 'Anecoica':
    archivo = np.genfromtxt('Database/Anecoica_regrabacion_d32cm/train-set_anecoica_32.txt', delimiter='; ', dtype='str', skip_header=1)
    os.chdir("Database/Anecoica_regrabacion_d32cm")
arch_length = np.size(archivo, 0)

''' An epoch describes the number of times the algorithm sees the entire data set. 
So, each time the algorithm has seen all samples in the dataset, an epoch has completed.

An iteration describes the number of times a batch of data passed through the algorithm. 
In the case of neural networks, that means the forward pass and backward pass. 
So, every time you pass a batch of data through the NN, you completed an iteration.
'''
cont = 0

error_cross_entropy_vec = np.arange(0)
for epoch in range(num_epochs):
    np.random.shuffle(archivo)
    print 'Epoch: '+str(epoch + 1)+' de '+str(num_epochs)
    for arch_ in range(arch_length):
        arch = arch_
        print 'Epoch: ' + str(epoch + 1) + '. Procesando archivo ' + str(arch + 1) + ' de ' + str(arch_length) + \
              '. alpha = ' + str(alpha_lnqt)
        cont += 1
        filename = archivo[arch, 0]
        g_tr_fname = archivo[arch, 1]
        in_sig, samp_freq, enc, n_samples = tools.read_audio(filename)
        #############################################################
        #                   Pre-processing                          #
        #############################################################
        if arch_ == 0:
            triang_filt_num, triang_filt_den = tools.triang_filters(f_inic, n_fft, n_quarter_tones, min_den_filt,
                                                                    samp_freq)
        supervector_matrix, time_vector = tools.get_supervector_matrix(in_sig, samp_freq, n_samples, win_length,
                                                                       overlap, n_fft, arch_, f_inic, n_quarter_tones,
                                                                       min_den_filt, alpha_lnqt, concatenated_frames,
                                                                       l_supervector, triang_filt_num, triang_filt_den)
        n_frames = np.size(supervector_matrix, 1)
        #############################################################
        #                   GROUND TRUTH                            #
        #############################################################
        ground_truth = tools.ground_truth_beatles(g_tr_fname, time_vector, n_frames)
        ############################################################
        #                  DNN - Train Stage                       #
        ############################################################
        ''' 
            Based by Korzeniowsky & Wieder:
                - 2670-dimensional input vector
                - 3 hidden-layers, 512-dimensional each.
                - minibatch: batch size: 512 for each epoch. ADAM update rule.
                - Dropout: 0.5 probability after hidden layer, stopping if validation accuracy not incresing after 
                    20 epochs.
            ESTÁN DEFINIDAS LAS VARIABLES.
            TO DO:
                - Run the DNN.
        '''
        n_iterations = int(float(n_frames)/batch_size)
        n_batchs = int(math.ceil(float(n_frames)/batch_size))
        supervector_matrix = np.float32(supervector_matrix)
        ground_truth = np.float32(ground_truth)
        for bch in range(n_batchs):
            if bch < n_batchs - 1:
                in_bch = bch * batch_size
                end_bch = (bch + 1) * batch_size - 1
                batch_inputs = supervector_matrix[:, in_bch:end_bch]
                batch_outputs = ground_truth[:, in_bch:end_bch]
            else:
                in_bch = bch * batch_size
                batch_inputs = supervector_matrix[:, in_bch:n_frames]
                batch_outputs = ground_truth[:, in_bch:n_frames]
            for iteration in range(n_iterations):
                train_step.run(feed_dict={in_layer: batch_inputs, out_gt: batch_outputs})
    ############################################################
    #                  DNN - Cross-validation                  #
    ############################################################
    '''
    W1_out = sess.run(W1)
    W2_out = sess.run(W2)
    W3_out = sess.run(W3)
    W4_out = sess.run(W4)
    b1_out = sess.run(b1)
    b2_out = sess.run(b2)
    b3_out = sess.run(b3)
    b4_out = sess.run(b4)
    '''
    os.chdir("..")
    if training_dataset == 'Original':
        validation_files = np.genfromtxt('Database/cross-val-set.txt', delimiter='; ', dtype='str', skip_header=1)
        os.chdir('Database')
    if training_dataset == 'Anecoica':
        validation_files = np.genfromtxt('Anecoica_regrabacion_d32cm/cross-val-set.txt', delimiter='; ', dtype='str', skip_header=1)
        os.chdir("Anecoica_regrabacion_d32cm")
    val_length = np.size(validation_files, 0)
    error_val = 0.0
    for val in range(val_length):
        filename = validation_files[val, 0]
        g_tr_fname = validation_files[val, 1]
        in_sig, samp_freq, enc, n_samples = tools.read_audio(filename)
        if arch_ == 0:
            triang_filt_num, triang_filt_den = tools.triang_filters(f_inic, n_fft, n_quarter_tones, min_den_filt,
                                                                    samp_freq)
        supervector_matrix, time_vector = tools.get_supervector_matrix(in_sig, samp_freq, n_samples, win_length,
                                                                       overlap, n_fft, arch_, f_inic,
                                                                       n_quarter_tones,
                                                                       min_den_filt, alpha_lnqt,
                                                                       concatenated_frames,
                                                                       l_supervector, triang_filt_num,
                                                                       triang_filt_den)
        n_frames = np.size(supervector_matrix, 1)
        ground_truth = tools.ground_truth_beatles(g_tr_fname, time_vector, n_frames)
        tic = time.clock()
        x_val = supervector_matrix
        error_net = tf.divide(-tf.reduce_sum(tf.add(tf.multiply(out_gt, tf.log(out_layer+eps)),
                                                    tf.multiply(1-out_gt+eps, tf.log(1-out_layer+eps)))),
                              tf.constant(value=12.0))
        error_val += error_net.eval(feed_dict={in_layer: supervector_matrix, out_gt: ground_truth}) / n_frames
    error_val = error_val / val_length
    learning[epoch] = error_val
    print "Error epoch "+str(epoch + 1)+': '+str(error_val)

############################################################
#                  DNN-Parameters Export                   #
############################################################

W1 = sess.run(W1)
W2 = sess.run(W2)
W3 = sess.run(W3)
W4 = sess.run(W4)
b1 = sess.run(b1)
b2 = sess.run(b2)
b3 = sess.run(b3)
b4 = sess.run(b4)
os.chdir("..")
if training_dataset == 'Anecoica':
    os.chdir("..")

np.savetxt('Parametros-DNN/W1_'+'epoch_'+str(num_epochs)+'_alpha_'+str(alpha_lnqt)+'_.out',
           W1, delimiter=',')
np.savetxt('Parametros-DNN/W2_'+'epoch_'+str(num_epochs)+'_alpha_'+str(alpha_lnqt)+'_.out',
           W2, delimiter=',')
np.savetxt('Parametros-DNN/W3_'+'epoch_'+str(num_epochs)+'_alpha_'+str(alpha_lnqt)+'_.out',
           W3, delimiter=',')
np.savetxt('Parametros-DNN/W4_'+'epoch_'+str(num_epochs)+'_alpha_'+str(alpha_lnqt)+'_.out',
           W4, delimiter=',')
np.savetxt('Parametros-DNN/b1_'+'epoch_'+str(num_epochs)+'_alpha_'+str(alpha_lnqt)+'_.out',
           b1, delimiter=',')
np.savetxt('Parametros-DNN/b2_'+'epoch_'+str(num_epochs)+'_alpha_'+str(alpha_lnqt)+'_.out',
           b2, delimiter=',')
np.savetxt('Parametros-DNN/b3_'+'epoch_'+str(num_epochs)+'_alpha_'+str(alpha_lnqt)+'_.out',
           b3, delimiter=',')
np.savetxt('Parametros-DNN/b4_'+'epoch_'+str(num_epochs)+'_alpha_'+str(alpha_lnqt)+'_.out',
           b4, delimiter=',')
sess.close()
#############################################################
#                   Mensaje final                           #
#############################################################
print("Done, my masta.")
