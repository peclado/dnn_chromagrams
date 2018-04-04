"""
Chroma Plots
Luis Alvarado (2017)
"""
import matplotlib.pyplot as plt
import math
import numpy as np

chromatic_scale = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
midi_b0 = 23


def pitch_plt(spec_mat, n_qt, time_vector, inic_chr, fin_chr, inic_frm, fin_frm):
    # n_qt: Number of quarter tones
    grid = False
    chr_dx = 11
    octava = 0
    midi_idx = 0
    chroma_scale = [None] * n_qt
    midi_note = [None] * n_qt
    time_vector = time_vector[inic_frm:fin_frm]
    for idx in range(n_qt):
        if math.fmod(idx, 2) == 1:
            chroma_scale[idx] = ' '
            midi_note[idx] = ' '
        else:
            chroma_scale[idx] = chromatic_scale[chr_dx] + str(octava)
            midi_note[idx] = str(midi_b0 + midi_idx)
            midi_idx += 1
            chr_dx += 1
            if chr_dx > 11:
                chr_dx = 0
                octava += 1
    #rango_frames = 120
    rango_chroma = np.arange(n_qt)
    rango_chroma = rango_chroma[::-1]


    ylab = [None] * n_qt
    # fin_chroma = 100
    for idx in range(n_qt):
        if math.fmod(idx, 2) == 0:
            ylab[idx] = chroma_scale[idx] + ' midi: ' + midi_note[idx]
        else:
            ylab[idx] = ' '
    midi_note = midi_note[inic_chr:fin_chr]
    rango_chroma = rango_chroma[inic_chr:fin_chr]
    chroma_scale = chroma_scale[inic_chr:fin_chr]
    ylab = ylab[inic_chr:fin_chr]

    '''
    f, axarr = plt.subplots(2, 2)
    for idx in range(4):
        alpha = float(idx)/3
        quarter_note_spectrogram_LN = np.divide(quarter_note_spectrogram_num, np.power(quarter_note_spectrogram_den, alpha))
        if idx == 0:
            axarr[0, 0].imshow(np.log(np.flipud(quarter_note_spectrogram_LN[inic_chroma:fin_chroma, 0:rango_frames])+1),
                               aspect='auto')
            axarr[0, 0].set_title('alpha = '+str(alpha))
            axarr[0, 0].set_yticks(range(len(rango_chroma)))
            axarr[0, 0].set_yticklabels(chroma_scale[::-1]+midi_note[::-1])
        if idx == 1:
            axarr[0, 1].imshow(np.log(np.flipud(quarter_note_spectrogram_LN[inic_chroma:fin_chroma, 0:rango_frames])+1),
                               aspect='auto')
            axarr[0, 1].set_title('alpha = '+str(alpha))
            axarr[0, 1].set_yticks(range(len(rango_chroma)))
            axarr[0, 1].set_yticklabels(chroma_scale[::-1])
        if idx == 2:
            axarr[1, 0].imshow(np.log(np.flipud(quarter_note_spectrogram_LN[inic_chroma:fin_chroma, 0:rango_frames])+1),
                               aspect='auto')
            axarr[1, 0].set_title('alpha = '+str(alpha))
            axarr[1, 0].set_yticks(range(len(rango_chroma)))
            axarr[1, 0].set_yticklabels(chroma_scale[::-1])
        if idx == 3:
            axarr[1, 1].imshow(np.log(np.flipud(quarter_note_spectrogram_LN[inic_chroma:fin_chroma, 0:rango_frames])+1),
                               aspect='auto')
            axarr[1, 1].set_title('alpha = '+str(alpha))
            axarr[1, 1].set_yticks(range(len(rango_chroma)))
            axarr[1, 1].set_yticklabels(chroma_scale[::-1])
    plt.show()
    '''
    plt.imshow(np.log(np.flipud(spec_mat[inic_chr:fin_chr, inic_frm:fin_frm]) + 1),
               aspect='auto')
    plt.colorbar()
    # plt.yticks(range(len(rango_chroma)), chroma_scale[::-1]+midi_note[::-1])
    plt.yticks(range(len(rango_chroma)), ylab[::-1])
    # time_idx = [0, len(time_vector) / 4 - 1, len(time_vector) / 2 - 1, 3 * len(time_vector) / 4 - 1,
                #len(time_vector) - 1]
    plt.xticks(range(len(time_vector)), np.round(time_vector,decimals=2))
    plt.grid(grid)
    # plt.show()


def chromagram_plot(chromagram, time_vector):
    plt.imshow(chromagram, aspect='auto', origin='origin')
    #plt.xticks(range(np.size(chromagram, 1)), time_vector)
    plt.xticks(time_vector)
    plt.yticks(range(12), ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'])
    # plt.show()

