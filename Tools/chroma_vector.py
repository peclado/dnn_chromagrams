import numpy as np
# chroma_vector = np.array([C(1),C#(2),D(3),D#(4),E(5),F(6),F#(7),G(8),G#(9),A(10),A#(11),B(12)]) # 12 valores


def flat_to_sharp(ch):  # ch: chord
    chord = ch
    if 'Cb' in ch:
        chord = 'B' + ch[2:]
        # ch.replace('Cb', 'B')
    if 'Db' in ch:
        chord = 'C#' + ch[2:]
        # ch.replace('Db', 'C#')
    if 'Eb' in ch:
        chord = 'D#' + ch[2:]
        # ch.replace('Eb', 'D#')
    if 'Fb' in ch:
        chord = 'E' + ch[2:]
        # ch.replace('Fb', 'E')
    if 'Gb' in ch:
        chord = 'F#' + ch[2:]
        # ch.replace('Gb', 'F#')
    if 'Ab' in ch:
        chord = 'G#' + ch[2:]
        # ch.replace('Ab', 'G#')
    if 'Bb' in ch:
        chord = 'A#' + ch[2:]
    return chord


def chroma_vector_beatles(ch):  # ch: chord
    global chr_vec

    """

    :type ch: object
    """
    eps = np.finfo(float).eps

    # Transformation from complex chords to maj/min
    if ch == 'C:maj7' or ch == 'C:maj(9)' or ch == 'C:7' or ch == 'C:7/3' or ch == 'C:9' or ch == 'C/2' or ch == 'C/3':
        ch = 'C'
    if ch == 'C:min(7)' or ch == 'C:min7':
        ch = 'C:min'
    if ch == 'C#:maj7' or ch == 'C#:maj(9)' or ch == 'C#:7' or ch == 'C#:7/3' or ch == 'C#:9' or ch == 'C#/2' or ch == 'C#/3':
        ch = 'C#'
    if ch == 'C#:min(7)' or ch == 'C#:min7':
        ch = 'C#:min'
    if ch == 'D:maj7' or ch == 'D:maj(9)' or ch == 'D:7' or ch == 'D:7/3' or ch == 'D:9' or ch == 'D/2' or ch == 'D/3':
        ch = 'D'
    if ch == 'D:min(7)' or ch == 'D:min7':
        ch = 'D:min'
    if ch == 'D#:maj7' or ch == 'D#:maj(9)' or ch == 'D#:7' or ch == 'D#:7/3' or ch == 'D#:9' or ch == 'D#/2' or ch == 'D#/3':
        ch = 'D#'
    if ch == 'D#:min(7)' or ch == 'D#:min7':
        ch = 'D#:min'
    if ch == 'E:maj7' or ch == 'E:maj(9)' or ch == 'E:7' or ch == 'E:7/3' or ch == 'E:9' or ch == 'E/2' or ch == 'E/3':
        ch = 'E'
    if ch == 'E:min(7)' or ch == 'E:min7':
        ch = 'E:min'
    if ch == 'F:maj7' or ch == 'F:maj(9)' or ch == 'F:7' or ch == 'F:7/3' or ch == 'F:9' or ch == 'F/2' or ch == 'F/3':
        ch = 'F'
    if ch == 'F:min(7)' or ch == 'F:min7':
        ch = 'F:min'
    if ch == 'F#:maj7' or ch == 'F#:maj(9)' or ch == 'F#:7' or ch == 'F#:7/3' or ch == 'F#:9' or ch == 'F#/2' or ch == 'F#/3':
        ch = 'F#'
    if ch == 'F#:min(7)' or ch == 'F#:min7':
        ch = 'F#:min'
    if ch == 'G:maj7' or ch == 'G:maj(9)' or ch == 'G:7' or ch == 'G:7/3' or ch == 'G:9' or ch == 'G/2' or ch == 'G/3':
        ch = 'G'
    if ch == 'G:min(7)' or ch == 'G:min7':
        ch = 'G:min'
    if ch == 'G#:maj7' or ch == 'G#:maj(9)' or ch == 'G#:7' or ch == 'G#:7/3' or ch == 'G#:9' or ch == 'G#/2' or ch == 'G#/3':
        ch = 'G#'
    if ch == 'G#:min(7)' or ch == 'G#:min7':
        ch = 'G#:min'
    if ch == 'A:maj7' or ch == 'A:maj(9)' or ch == 'A:7' or ch == 'A:7/3' or ch == 'A:9' or ch == 'A/2' or ch == 'A/3':
        ch = 'A'
    if ch == 'A:min(7)' or ch == 'A:min7' or ch == 'A:min/b3':
        ch = 'A:min'
    if ch == 'A#:maj7' or ch == 'A#:maj(9)' or ch == 'A#:7' or ch == 'A#:7/3' or ch == 'A#:9' or ch == 'A#/2' or ch == 'A#/3':
        ch = 'A#'
    if ch == 'A#:min(7)' or ch == 'A#:min7':
        ch = 'A#:min'
    if ch == 'B:maj7' or ch == 'B:maj(9)' or ch == 'B:7' or ch == 'B:7/3' or ch == 'B:9' or ch == 'B/2' or ch == 'B/3':
        ch = 'B'
    if ch == 'B:min(7)' or ch == 'B:min7':
        ch = 'B:min'
    # Transformation from complex to aug/dim chords.
    if ch == 'C:aug(7)':
        ch = 'C:aug'
    if ch == 'C:dim(7)' or ch == 'C:hdim(7)' or ch == 'C:dim7':
        ch = 'C:dim'
    if ch == 'C#:aug(7)':
        ch = 'C#:aug'
    if ch == 'C#:dim(7)' or ch == 'C#:hdim(7)' or ch == 'C#:dim7':
        ch = 'C#:dim'
    if ch == 'D:aug(7)':
        ch = 'D:aug'
    if ch == 'D:dim(7)' or ch == 'D:hdim(7)' or ch == 'D:dim7':
        ch = 'D:dim'
    if ch == 'D#:aug(7)':
        ch = 'D#:aug'
    if ch == 'D#:dim(7)' or ch == 'D#:hdim(7)' or ch == 'D#:dim7':
        ch = 'D#:dim'
    if ch == 'E:aug(7)':
        ch = 'E:aug'
    if ch == 'E:dim(7)' or ch == 'E:hdim(7)' or ch == 'E:dim7':
        ch = 'E:dim'
    if ch == 'F:aug(7)':
        ch = 'F:aug'
    if ch == 'F:dim(7)' or ch == 'F:hdim(7)' or ch == 'F:dim7':
        ch = 'F:dim'
    if ch == 'F#:aug(7)':
        ch = 'F#:aug'
    if ch == 'F#:dim(7)' or ch == 'F#:hdim(7)' or ch == 'F#:dim7':
        ch = 'F#:dim'
    if ch == 'G:aug(7)':
        ch = 'G:aug'
    if ch == 'G:dim(7)' or ch == 'G:hdim(7)' or ch == 'G:dim7':
        ch = 'G:dim'
    if ch == 'G#:aug(7)':
        ch = 'G#:aug'
    if ch == 'G#:dim(7)' or ch == 'G#:hdim(7)' or ch == 'G#:dim7':
        ch = 'G#:dim'
    if ch == 'A:aug(7)':
        ch = 'A:aug'
    if ch == 'A:dim(7)' or ch == 'A:hdim(7)' or ch == 'A:dim7':
        ch = 'A:dim'
    if ch == 'A#:aug(7)':
        ch = 'A#:aug'
    if ch == 'A#:dim(7)' or ch == 'A#:hdim(7)' or ch == 'A#:dim7':
        ch = 'A#:dim'
    if ch == 'B:aug(7)':
        ch = 'B:aug'
    if ch == 'B:dim(7)' or ch == 'B:hdim(7)' or ch == 'B:dim7':
        ch = 'B:dim'
    # Transformation from complex to suspended chords
    if ch == 'C:sus4(7)' or ch == 'C:sus4(b7)':
        ch = 'C:sus4'
    if ch == 'C#:sus4(7)' or ch == 'C#:sus4(b7)':
        ch = 'C#:sus4'
    if ch == 'D:sus4(7)' or ch == 'D:sus4(b7)':
        ch = 'D:sus4'
    if ch == 'D#:sus4(7)' or ch == 'D#:sus4(b7)':
        ch = 'D#:sus4'
    if ch == 'E:sus4(7)' or ch == 'E:sus4(b7)':
        ch = 'E:sus4'
    if ch == 'F:sus4(7)' or ch == 'F:sus4(b7)':
        ch = 'F:sus4'
    if ch == 'F#:sus4(7)' or ch == 'F#:sus4(b7)':
        ch = 'F#:sus4'
    if ch == 'G:sus4(7)' or ch == 'G:sus4(b7)':
        ch = 'G:sus4'
    if ch == 'G#:sus4(7)' or ch == 'G#:sus4(b7)':
        ch = 'G#:sus4'
    if ch == 'A:sus4(7)' or ch == 'A:sus4(b7)':
        ch = 'A:sus4'
    if ch == 'A#:sus4(7)' or ch == 'A#:sus4(b7)':
        ch = 'A#:sus4'
    if ch == 'B:sus4(7)' or ch == 'B:sus4(b7)':
        ch = 'B:sus4'

    # Selecting chroma vectors from chord label
    if ch == 'N':  # No chord
        chr_vec = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    # Majors
    elif ch == 'C' or ch == 'C:maj':  # C major
        chr_vec = np.array([1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0])
    elif ch == 'C#' or ch == 'C#:maj':  # C# major
        chr_vec = np.array([0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0])
    elif ch == 'D' or ch == 'D:maj': # D major
        chr_vec = np.array([0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0])
    elif ch == 'D#' or ch == 'D#:maj':  # D# major
        chr_vec = np.array([0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0])
    elif ch == 'E' or ch == 'E:maj':   # E major
        chr_vec = np.array([0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1])
    elif ch == 'F' or ch == 'F:maj':   # F major
        chr_vec = np.array([1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0])
    elif ch == 'F#' or ch == 'F#:maj':  # F# major
        chr_vec = np.array([0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0])
    elif ch == 'G' or ch == 'G:maj':  # G major
        chr_vec = np.array([0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1])
    elif ch == 'G#' or ch == 'G#:maj':  # G# major
        chr_vec = np.array([1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0])
    elif ch == 'A' or ch == 'A:maj':  # A major
        chr_vec = np.array([0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0])
    elif ch == 'A#' or ch == 'A#:maj': # A# major
        chr_vec = np.array([0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0])
    elif ch == 'B' or ch == 'B:maj':  # B major
        chr_vec = np.array([0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1])
    # Minors
    elif ch == 'C:min':  # C minor
        chr_vec = np.array([1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0])
    elif ch == 'C#:min':  # C# minor
        chr_vec = np.array([0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0])
    elif ch == 'D:min':  # D minor
        chr_vec = np.array([0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0])
    elif ch == 'D#:min':  # D# minor
        chr_vec = np.array([0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0])
    elif ch == 'E:min':  # E minor
        chr_vec = np.array([0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1])
    elif ch == 'F:min':  # F minor
        chr_vec = np.array([1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0])
    elif ch == 'F#:min':  # F# minor
        chr_vec = np.array([0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0])
    elif ch == 'G:min':  # G minor
        chr_vec = np.array([0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0])
    elif ch == 'G#:min':  # G# minor
        chr_vec = np.array([0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1])
    elif ch == 'A:min':  # A minor
        chr_vec = np.array([1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0])
    elif ch == 'A#:min':  # A# minor
        chr_vec = np.array([0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0])
    elif ch == 'B:min':  # B minor
        chr_vec = np.array([0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1])
    # Augmented
    elif ch == 'C:aug':  # C augmented
        chr_vec = np.array([1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0])
    elif ch == 'C#:aug':  # C# augmented
        chr_vec = np.array([0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0])
    elif ch == 'D:aug': # D augmented
        chr_vec = np.array([0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0])
    elif ch == 'D#:aug':  # D# augmented
        chr_vec = np.array([0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1])
    elif ch == 'E:aug':   # E augmented
        chr_vec = np.array([1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0])
    elif ch == 'F:aug':   # F augmented
        chr_vec = np.array([0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0])
    elif ch == 'F#:aug':  # F# augmented
        chr_vec = np.array([0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0])
    elif ch == 'G:aug':  # G augmented
        chr_vec = np.array([0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1])
    elif ch == 'G#:aug':  # G# augmented
        chr_vec = np.array([1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0])
    elif ch == 'A:aug':  # A augmented
        chr_vec = np.array([0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0])
    elif ch == 'A#:aug': # A# augmented
        chr_vec = np.array([0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0])
    elif ch == 'B:aug':  # B augmented
        chr_vec = np.array([0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1])
    # Diminished
    elif ch == 'C:dim':  # C diminished
        chr_vec = np.array([1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0])
    elif ch == 'C#:dim':  # C# diminished
        chr_vec = np.array([0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0])
    elif ch == 'D:dim':  # D diminished
        chr_vec = np.array([0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0])
    elif ch == 'D#:dim':  # D# diminished
        chr_vec = np.array([0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0])
    elif ch == 'E:dim':  # E diminished
        chr_vec = np.array([0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0])
    elif ch == 'F:dim':  # F diminished
        chr_vec = np.array([0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1])
    elif ch == 'F#:dim':  # F# diminished
        chr_vec = np.array([1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0])
    elif ch == 'G:dim':  # G diminished
        chr_vec = np.array([0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0])
    elif ch == 'G#:dim':  # G# diminished
        chr_vec = np.array([0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1])
    elif ch == 'A:dim':  # A diminished
        chr_vec = np.array([1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0])
    elif ch == 'A#:dim':  # A# diminished
        chr_vec = np.array([0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0])
    elif ch == 'B:dim':  # B diminished
        chr_vec = np.array([0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1])
    # Extended Chords
    elif ch == 'C:sus2':  # C suspended 2th
        chr_vec = np.array([1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0])
    elif ch == 'C#:sus2':  # C# suspended 2th
        chr_vec = np.array([0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0])
    elif ch == 'D:sus2':  # D suspended 2th
        chr_vec = np.array([0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0])
    elif ch == 'D#:sus2':  # D# suspended 2th
        chr_vec = np.array([0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0])
    elif ch == 'E:sus2':  # E suspended 2th
        chr_vec = np.array([0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1])
    elif ch == 'F:sus2':  # F suspended 2th
        chr_vec = np.array([1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0])
    elif ch == 'F#:sus2':  # F# suspended 2th
        chr_vec = np.array([0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0])
    elif ch == 'G:sus2':  # G suspended 2th
        chr_vec = np.array([0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0])
    elif ch == 'G#:sus2':  # G# suspended 2th
        chr_vec = np.array([0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0])
    elif ch == 'A:sus2':  # A suspended 2th
        chr_vec = np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1])
    elif ch == 'A#:sus2':  # A# suspended 2th
        chr_vec = np.array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0])
    elif ch == 'B:sus2':  # B suspended 2th
        chr_vec = np.array([0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1])
    elif ch == 'C:sus4':  # C suspended 4th
        chr_vec = np.array([1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0])
    elif ch == 'C#:sus4':  # C# suspended 4th
        chr_vec = np.array([0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0])
    elif ch == 'D:sus4':  # D suspended 4th
        chr_vec = np.array([0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0])
    elif ch == 'D#:sus4':  # D# suspended 4th
        chr_vec = np.array([0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0])
    elif ch == 'E:sus4':  # E suspended 4th
        chr_vec = np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1])
    elif ch == 'F:sus4':  # F suspended 4th
        chr_vec = np.array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0])
    elif ch == 'F#:sus4':  # F# suspended 4th
        chr_vec = np.array([0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1])
    elif ch == 'G:sus4':  # G suspended 4th
        chr_vec = np.array([1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0])
    elif ch == 'G#:sus4':  # G# suspended 4th
        chr_vec = np.array([0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0])
    elif ch == 'A:sus4':  # A suspended 4th
        chr_vec = np.array([0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0])
    elif ch == 'A#:sus4':  # A# suspended 4th
        chr_vec = np.array([0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0])
    elif ch == 'B:sus4':  # B suspended 4th
        chr_vec = np.array([0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1])
    else:
        'No se encontro el acorde: ' + ch
    # chr_vec = np.divide(chr_vec, np.sum(chr_vec) + eps)
    return chr_vec


def get_out_chord(y):
    if not y.any():
        out_chord = 'N'
    # Majors
    elif np.array_equal(y, np.array([[0, 4, 7]])):
        out_chord = 'C:maj'
    elif np.array_equal(y, np.array([[1, 5, 8]])):
        out_chord = 'C#:maj'
    elif np.array_equal(y, np.array([[2, 6, 9]])):
        out_chord = 'D:maj'
    elif np.array_equal(y, np.array([[3, 7, 10]])):
        out_chord = 'D#:maj'
    elif np.array_equal(y, np.array([[4, 8, 11]])):
        out_chord = 'E:maj'
    elif np.array_equal(y, np.array([[0, 5, 9]])):
        out_chord = 'F:maj'
    elif np.array_equal(y, np.array([[1, 6, 10]])):
        out_chord = 'F#:maj'
    elif np.array_equal(y, np.array([[2, 7, 11]])):
        out_chord = 'G:maj'
    elif np.array_equal(y, np.array([[0, 3, 8]])):
        out_chord = 'G#:maj'
    elif np.array_equal(y, np.array([[1, 4, 9]])):
        out_chord = 'A:maj'
    elif np.array_equal(y, np.array([[2, 5, 10]])):
        out_chord = 'A#:maj'
    elif np.array_equal(y, np.array([[3, 6, 11]])):
        out_chord = 'B:maj'
    # Minors
    elif np.array_equal(y, np.array([[0, 3, 7]])):
        out_chord = 'C:min'
    elif np.array_equal(y, np.array([[1, 4, 8]])):
        out_chord = 'C#:min'
    elif np.array_equal(y, np.array([[2, 5, 9]])):
        out_chord = 'D:min'
    elif np.array_equal(y, np.array([[3, 6, 10]])):
        out_chord = 'D#:min'
    elif np.array_equal(y, np.array([[4, 7, 11]])):
        out_chord = 'E:min'
    elif np.array_equal(y, np.array([[0, 5, 8]])):
        out_chord = 'F:min'
    elif np.array_equal(y, np.array([[1, 6, 9]])):
        out_chord = 'F#:min'
    elif np.array_equal(y, np.array([[2, 7, 10]])):
        out_chord = 'G:min'
    elif np.array_equal(y, np.array([[3, 8, 11]])):
        out_chord = 'G#:min'
    elif np.array_equal(y, np.array([[0, 4, 9]])):
        out_chord = 'A:min'
    elif np.array_equal(y, np.array([[1, 5, 10]])):
        out_chord = 'A#:min'
    elif np.array_equal(y, np.array([[2, 6, 11]])):
        out_chord = 'B:min'
    # Augmented
    elif np.array_equal(y, np.array([[0, 4, 8]])):
        out_chord = 'C:aug'
    elif np.array_equal(y, np.array([[1, 5, 9]])):
        out_chord = 'C#:aug'
    elif np.array_equal(y, np.array([[2, 6, 10]])):
        out_chord = 'D:aug'
    elif np.array_equal(y, np.array([[3, 7, 11]])):
        out_chord = 'D#:aug'
    elif np.array_equal(y, np.array([[0, 4, 8]])):
        out_chord = 'E:aug'
    elif np.array_equal(y, np.array([[1, 5, 9]])):
        out_chord = 'F:aug'
    elif np.array_equal(y, np.array([[2, 6, 10]])):
        out_chord = 'F#:aug'
    elif np.array_equal(y, np.array([[3, 7, 11]])):
        out_chord = 'G:aug'
    elif np.array_equal(y, np.array([[0, 4, 8]])):
        out_chord = 'G#:aug'
    elif np.array_equal(y, np.array([[1, 5, 9]])):
        out_chord = 'A:aug'
    elif np.array_equal(y, np.array([[2, 6, 10]])):
        out_chord = 'A#:aug'
    elif np.array_equal(y, np.array([[3, 7, 11]])):
        out_chord = 'B:aug'
    # Diminished
    elif np.array_equal(y, np.array([[0, 3, 6]])):
        out_chord = 'C:dim'
    elif np.array_equal(y, np.array([[1, 4, 7]])):
        out_chord = 'C#:dim'
    elif np.array_equal(y, np.array([[2, 5, 8]])):
        out_chord = 'D:dim'
    elif np.array_equal(y, np.array([[3, 6, 9]])):
        out_chord = 'D#:dim'
    elif np.array_equal(y, np.array([[4, 7, 10]])):
        out_chord = 'E:dim'
    elif np.array_equal(y, np.array([[5, 8, 11]])):
        out_chord = 'F:dim'
    elif np.array_equal(y, np.array([[0, 6, 9]])):
        out_chord = 'F#:dim'
    elif np.array_equal(y, np.array([[1, 7, 10]])):
        out_chord = 'G:dim'
    elif np.array_equal(y, np.array([[2, 8, 11]])):
        out_chord = 'G#:dim'
    elif np.array_equal(y, np.array([[0, 3, 9]])):
        out_chord = 'A:dim'
    elif np.array_equal(y, np.array([[1, 4, 10]])):
        out_chord = 'A#:dim'
    elif np.array_equal(y, np.array([[2, 5, 11]])):
        out_chord = 'B:dim'
    # Suspended 2th
    elif np.array_equal(y, np.array([[0, 2, 7]])):
        out_chord = 'C:sus2'
    elif np.array_equal(y, np.array([[1, 3, 8]])):
        out_chord = 'C#:sus2'
    elif np.array_equal(y, np.array([[2, 4, 9]])):
        out_chord = 'D:sus2'
    elif np.array_equal(y, np.array([[3, 5, 10]])):
        out_chord = 'D#:sus2'
    elif np.array_equal(y, np.array([[4, 6, 11]])):
        out_chord = 'E:sus2'
    elif np.array_equal(y, np.array([[0, 5, 7]])):
        out_chord = 'F:sus2'
    elif np.array_equal(y, np.array([[1, 6, 8]])):
        out_chord = 'F#:sus2'
    elif np.array_equal(y, np.array([[2, 7, 9]])):
        out_chord = 'G:sus2'
    elif np.array_equal(y, np.array([[3, 8, 10]])):
        out_chord = 'G#:sus2'
    elif np.array_equal(y, np.array([[4, 9, 11]])):
        out_chord = 'A:sus2'
    elif np.array_equal(y, np.array([[0, 5, 10]])):
        out_chord = 'A#:sus2'
    elif np.array_equal(y, np.array([[1, 6, 11]])):
        out_chord = 'B:sus2'
    # Suspended 4th
    elif np.array_equal(y, np.array([[0, 5, 7]])):
        out_chord = 'C:sus4'
    elif np.array_equal(y, np.array([[1, 6, 8]])):
        out_chord = 'C#:sus4'
    elif np.array_equal(y, np.array([[2, 7, 9]])):
        out_chord = 'D:sus4'
    elif np.array_equal(y, np.array([[3, 8, 10]])):
        out_chord = 'D#:sus4'
    elif np.array_equal(y, np.array([[4, 9, 11]])):
        out_chord = 'E:sus4'
    elif np.array_equal(y, np.array([[0, 5, 10]])):
        out_chord = 'F:sus4'
    elif np.array_equal(y, np.array([[1, 6, 11]])):
        out_chord = 'F#:sus4'
    elif np.array_equal(y, np.array([[0, 2, 7]])):
        out_chord = 'G:sus4'
    elif np.array_equal(y, np.array([[1, 3, 8]])):
        out_chord = 'G#:sus4'
    elif np.array_equal(y, np.array([[2, 4, 9]])):
        out_chord = 'A:sus4'
    elif np.array_equal(y, np.array([[3, 5, 10]])):
        out_chord = 'A#:sus4'
    elif np.array_equal(y, np.array([[4, 6, 11]])):
        out_chord = 'B:sus4'
    else:
        out_chord = 'N'
    return out_chord


def chord_for_comparison():
    N = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    C_maj = np.array([[1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0]])
    C_sh_maj = np.array([[0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0]])
    D_maj = np.array([[0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0]])
    D_sh_maj = np.array([[0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0]])
    E_maj = np.array([[0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1]])
    F_maj = np.array([[1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0]])
    F_sh_maj = np.array([[0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0]])
    G_maj = np.array([[0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1]])
    G_sh_maj = np.array([[1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0]])
    A_maj= np.array([[0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0]])
    A_sh_maj = np.array([[0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0]])
    B_maj = np.array([[0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1]])
    # Minors
    C_min = np.array([[1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0]])
    C_sh_min = np.array([[0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]])
    D_min = np.array([[0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0]])
    D_sh_min = np.array([[0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0]])
    E_min = np.array([[0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1]])
    F_min = np.array([[1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0]])
    F_sh_min = np.array([[0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0]])
    G_min = np.array([[0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0]])
    G_sh_min = np.array([[0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1]])
    A_min = np.array([[1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0]])
    A_sh_min = np.array([[0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0]])
    B_min = np.array([[0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1]])
    # Augmented
    C_aug = np.array([[1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]])
    C_sh_aug = np.array([[0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0]])
    D_aug = np.array([[0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0]])
    D_sh_aug = np.array([[0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1]])
    E_aug = np.array([[1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]])
    F_aug = np.array([[0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0]])
    F_sh_aug = np.array([[0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0]])
    G_aug = np.array([[0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1]])
    G_sh_aug = np.array([[1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]])
    A_aug = np.array([[0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0]])
    A_sh_aug = np.array([[0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0]])
    B_aug = np.array([[0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1]])
    # Diminished
    C_dim = np.array([[1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0]])
    C_sh_dim = np.array([[0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0]])
    D_dim = np.array([[0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0]])
    D_sh_dim = np.array([[0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0]])
    E_dim = np.array([[0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0]])
    F_dim = np.array([[0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1]])
    F_sh_dim = np.array([[1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0]])
    G_dim = np.array([[0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0]])
    G_sh_dim = np.array([[0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1]])
    A_dim = np.array([[1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0]])
    A_sh_dim = np.array([[0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0]])
    B_dim = np.array([[0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1]])
    # Extended Chords
    C_sus2 = np.array([[1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0]])
    C_sh_sus2 = np.array([[0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0]])
    D_sus2 = np.array([[0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0]])
    D_sh_sus2 = np.array([[0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0]])
    E_sus2 = np.array([[0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1]])
    F_sus2 = np.array([[1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0]])
    F_sh_sus2 = np.array([[0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0]])
    G_sus2 = np.array([[0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0]])
    G_sh_sus2 = np.array([[0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0]])
    A_sus2 = np.array([[0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1]])
    A_sh_sus2 = np.array([[1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0]])
    B_sus2 = np.array([[0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]])
    C_sus4 = np.array([[1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0]])
    C_sh_sus4 = np.array([[0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0]])
    D_sus4 = np.array([[0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0]])
    D_sh_sus4 = np.array([[0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0]])
    E_sus4 = np.array([[0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1]])
    F_sus4 = np.array([[1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0]])
    F_sh_sus4 = np.array([[0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]])
    G_sus4 = np.array([[1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0]])
    G_sh_sus4 = np.array([[0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0]])
    A_sus4 = np.array([[0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0]])
    A_sh_sus4 = np.array([[0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0]])
    B_sus4= np.array([[0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1]])
    chord_for_comp = np.concatenate((N.T, C_maj.T, C_sh_maj.T, D_maj.T, D_sh_maj.T, E_maj.T, F_maj.T, F_sh_maj.T,
                                     G_maj.T, G_sh_maj.T, A_maj.T, A_sh_maj.T, B_maj.T, C_min.T, C_sh_min.T, D_min.T,
                                     D_sh_min.T, E_min.T, F_min.T, F_sh_min.T, G_min.T, G_sh_min.T, A_min.T, A_sh_min.T,
                                     B_min.T), axis=1)
    return chord_for_comp


def get_chord(chord_for_comp):
        out_pb = ["N", "C:maj", "C#:maj", "D:maj", "D#:maj", "E:maj", "F:maj", "F#:maj", "G:maj", "G#:maj", "A:maj",
                  "A#:maj", "B:maj", "C:min", "C#:min", "D:min", "D#:min", "E:min", "F:min", "F#:min", "G:min",
                  "G#:min", "A:min", "A#:min", "B:min"]
        out_chr = []
        for i in range(len(chord_for_comp)):
            out_chr.append(out_pb[int(chord_for_comp[i])])
        '''
        if chord_for_comp == 0:
            out_ch = "N"
        elif chord_for_comp == 1:
            out_ch = "C:maj"
        elif chord_for_comp == 2:
            out_ch = "C#:maj"
        elif chord_for_comp == 3:
            out_ch = "D:maj"
        elif chord_for_comp == 4:
            out_ch = "D#:maj"
        elif chord_for_comp == 5
        '''
        return out_chr





