import math
import numpy as np


def audio2frame(signal,frame_length,frame_step,winfunc=lambda x:np.ones((x,))):
    """ Framing audio signal. Uses numbers of samples as unit.
    Args:
    signal: 1-D numpy array.
	frame_length: In this situation, frame_length=samplerate*win_length, since we
        use numbers of samples as unit.
    frame_step:In this situation, frame_step=samplerate*win_step,
        representing the number of samples between the start point of adjacent frames.
	winfunc:lambda function, to generate a vector with shape (x,) filled with ones.
    Returns:
        frames*win: 2-D numpy array with shape (frames_num, frame_length).
    """
    signal_length=len(signal)
    # Use round() to ensure length and step are integer, considering that we use numbers
    # of samples as unit.
    frame_length=int(round(frame_length))
    frame_step=int(round(frame_step))
    if signal_length<=frame_length:
        frames_num=1
    else:
        frames_num=1+int(math.ceil((1.0*signal_length-frame_length)/frame_step))
    pad_length=int((frames_num-1)*frame_step+frame_length)
    # Padding zeros at the end of signal if pad_length > signal_length.
    zeros=np.zeros((pad_length-signal_length,))
    pad_signal=np.concatenate((signal,zeros))
    # Calculate the indice of signal for every sample in frames, shape (frams_nums, frams_length)
    indices=np.tile(np.arange(0,frame_length),(frames_num,1))+np.tile(
        np.arange(0,frames_num*frame_step,frame_step),(frame_length,1)).T
    indices=np.array(indices,dtype=np.int32)
    # Get signal data according to indices.
    frames=pad_signal[indices]
    win=np.tile(winfunc(frame_length),(frames_num,1))
    return frames*win

def spectrum_magnitude(frames,NFFT):
    '''Apply FFT and Calculate magnitude of the spectrum.
    Args:
        frames: 2-D frames array calculated by audio2frame(...).
        NFFT:FFT size.
    Returns:
        Return magnitude of the spectrum after FFT, with shape (frames_num, NFFT).
    '''
    complex_spectrum=np.fft.rfft(frames,NFFT)
    return np.absolute(complex_spectrum)

def spectrum_power(frames,NFFT):
    """Calculate power spectrum for every frame after FFT.
    Args:
        frames: 2-D frames array calculated by audio2frame(...).
        NFFT:FFT size
    Returns:
        Power spectrum: PS = magnitude^2/NFFT
    """
    return 1.0/NFFT * np.square(spectrum_magnitude(frames,NFFT))

def log_spectrum_power(frames,NFFT,norm=1):
    '''Calculate log power spectrum.
    Args:
        frames:2-D frames array calculated by audio2frame(...)
        NFFT：FFT size
        norm: Norm.
    '''
    spec_power=spectrum_power(frames,NFFT)
    # In case of calculating log0, we set 0 in spec_power to 0.
    spec_power[spec_power<1e-30]=1e-30
    log_spec_power=10*np.log10(spec_power)
    if norm:
        return log_spec_power-np.max(log_spec_power)
    else:
        return log_spec_power

def pre_emphasis(signal,coefficient=0.95):
    '''Pre-emphasis.
    Args:
        signal: 1-D numpy array.
        coefficient:Coefficient for pre-emphasis. Defauted to 0.95.
    Returns:
        pre-emphasis signal.
    '''
    return np.append(signal[0],signal[1:]-coefficient*signal[:-1])


def list_to_sparse_tensor(targetList, level):
    ''' 
    turn 2-D List to SparseTensor
    '''
    indices = [] #index
    vals = [] #value
    assert level == 'phn' or level == 'cha', 'type must be phoneme or character, seq2seq will be supported in future'
    phn = ['aa',  'ae', 'ah',  'ao',  'aw', 'ax',  'ax-h',\
           'axr', 'ay', 'b',   'bcl', 'ch', 'd',   'dcl',\
           'dh',  'dx', 'eh',  'el',  'em', 'en',  'eng',\
           'epi', 'er', 'ey',  'f',   'g',  'gcl', 'h#',\
           'hh',  'hv', 'ih',  'ix',  'iy', 'jh',  'k',\
           'kcl', 'l',  'm',   'n',   'ng', 'nx',  'ow',\
           'oy',  'p',  'pau', 'pcl', 'q',  'r',   's',\
           'sh',  't',  'tcl', 'th',  'uh', 'uw',  'ux',\
           'v',   'w',  'y',   'z',   'zh']

    mapping = {'ah': 'ax', 'ax-h': 'ax', 'ux': 'uw', 'aa': 'ao', 'ih': 'ix', \
               'axr': 'er', 'el': 'l', 'em': 'm', 'en': 'n', 'nx': 'n',\
               'eng': 'ng', 'sh': 'zh', 'hv': 'hh', 'bcl': 'h#', 'pcl': 'h#',\
               'dcl': 'h#', 'tcl': 'h#', 'gcl': 'h#', 'kcl': 'h#',\
               'q': 'h#', 'epi': 'h#', 'pau': 'h#'}

    group_phn = ['ae', 'ao', 'aw', 'ax', 'ay', 'b', 'ch', 'd', 'dh', 'dx', 'eh', \
                 'er', 'ey', 'f', 'g', 'h#', 'hh', 'ix', 'iy', 'jh', 'k', 'l', \
                 'm', 'n', 'ng', 'ow', 'oy', 'p', 'r', 's', 't', 'th', 'uh', 'uw',\
                 'v', 'w', 'y', 'z', 'zh']


    mapping = {'ah': 'ax', 'ax-h': 'ax', 'ux': 'uw', 'aa': 'ao', 'ih': 'ix', \
               'axr': 'er', 'el': 'l', 'em': 'm', 'en': 'n', 'nx': 'n',\
               'eng': 'ng', 'sh': 'zh', 'hv': 'hh', 'bcl': 'h#', 'pcl': 'h#',\
               'dcl': 'h#', 'tcl': 'h#', 'gcl': 'h#', 'kcl': 'h#',\
               'q': 'h#', 'epi': 'h#', 'pau': 'h#'}

    group_phn = ['ae', 'ao', 'aw', 'ax', 'ay', 'b', 'ch', 'd', 'dh', 'dx', 'eh', \
                 'er', 'ey', 'f', 'g', 'h#', 'hh', 'ix', 'iy', 'jh', 'k', 'l', \
                 'm', 'n', 'ng', 'ow', 'oy', 'p', 'r', 's', 't', 'th', 'uh', 'uw',\
                 'v', 'w', 'y', 'z', 'zh']

    if level == 'cha':
        for tI, target in enumerate(targetList):
            for seqI, val in enumerate(target):
                indices.append([tI, seqI])
                vals.append(val)
        shape = [len(targetList), np.asarray(indices).max(axis=0)[1]+1] #shape
        return (np.array(indices), np.array(vals), np.array(shape))

    elif level == 'phn':
        '''
        for phn level, we should collapse 61 labels into 39 labels before scoring

        Reference:
          - Heterogeneous Acoustic Measurements and Multiple Classifiers for Speech Recognition(1986), 
          - Andrew K. Halberstadt, https://groups.csail.mit.edu/sls/publications/1998/phdthesis-drew.pdf
        '''
        for tI, target in enumerate(targetList):
            for seqI, val in enumerate(target):
                if val < len(phn) and (phn[val] in mapping.keys()):
                    val = group_phn.index(mapping[phn[val]])
                indices.append([tI, seqI])
                vals.append(val)
        shape = [len(targetList), np.asarray(indices).max(0)[1]+1] #shape
        return (np.array(indices), np.array(vals), np.array(shape))

    else:
        ##support seq2seq in future here
        raise ValueError('Invalid level: %s'%str(level))
