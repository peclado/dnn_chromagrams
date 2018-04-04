#!/usr/bin/env python
# -*- coding: utf-8 -*-

a = 3
'''
midi_b0 = 23
for jdx in range(n_frames):
    print ''
    for idx in range(np.size(g_tr_in, 0)):
        a = g_tr_in[idx, 0]
        t = time_vector[jdx]
        b = g_tr_in[idx, 1]
        if((a <= t) & (b >= t)):
            g_tr_idx = g_tr_in[idx, 2]
            g_tr_mat[2*(int(g_tr_idx)-midi_b0), jdx] = 1.0

'''
#############################################################
#                    Ground-truth                           #
#############################################################
'''
    GROUND TRUTH MAPSS
g_tr_in = np.loadtxt(g_tr_fname, skiprows=1)  # Ground-truth input matrix. Elimina texto indeseado.
g_tr_mat = np.zeros(shape=(n_quarter_tones, n_frames))
midi_b0 = 23
for jdx in range(n_frames):
    print ''
    for idx in range(np.size(g_tr_in, 0)):
        a = g_tr_in[idx, 0]
        t = time_vector[jdx]
        b = g_tr_in[idx, 1]
        if((a <= t) & (b >= t)):
            g_tr_idx = g_tr_in[idx, 2]
            g_tr_mat[2*(int(g_tr_idx)-midi_b0), jdx] = 1.0

'''
""" TODO: Generar una matriz de ground truth:
- Saber cuál es el tiempo de un frame.
- Dar valores a los frames no indicados.
- Dejar matriz en términos del frame y no en segundos.
"""

# print g_tr_mat



############################################################
#                       Plots                              #
############################################################
inic_chroma = 0
fin_chroma = n_quarter_tones
inic_frm = 0
fin_frm = n_frames
plt.figure(1)
# chroma_plt.pitch_plt(g_tr_mat, n_quarter_tones, time_vector, inic_chroma, fin_chroma, inic_frm,
#                     fin_frm)
# plt.figure(2)
chroma_plt.pitch_plt(quarter_note_spectrogram_num, n_quarter_tones, time_vector, inic_chroma, fin_chroma,
                     inic_frm, fin_frm)
plt.show()

#############################################################
#                   Plots                                   #
#############################################################
'''
time = np.linspace(1/samp_freq, float(n_samples)/samp_freq, n_samples)
plt.figure(1)
plt.plot(time[0:samp_freq*30-1], in_sig[0:samp_freq*30-1])
#plt.plot(in_sig_mag[0:n_fft/8-1, 500])
plt.xlabel("Tiempo en segundos")
#plt.ylabel("Amplitud")
#plt.plot(np.abs(in_sig_stft[:, 50]))
#plt.imshow(in_sig_mag[0:n_fft/4, 0:5000])
x_lab, y_lab = np.meshgrid(np.linspace(1/samp_freq, 30, 999), np.linspace(0, samp_freq/8, n_fft/8))
mat_plot = np.log10(in_sig_stft_mag[0:n_fft/8, 0:1000-1])
print np.size(x_lab, 0), np.size(x_lab, 1)
print np.size(y_lab, 0), np.size(y_lab, 1)
print np.size(mat_plot, 0), np.size(mat_plot, 1)
fig, ax = plt.subplots(figsize=(6, 6))
plt.pcolormesh(x_lab, y_lab, mat_plot)
plt.colorbar()
plt.xlabel("Tiempo en segundos")
plt.ylabel("Frecuencia en Hz")
'''

# ax.imshow(hist, cmap=plt.cm.Reds, interpolation='none', extent=[80,120,32,0])
plt.show()
