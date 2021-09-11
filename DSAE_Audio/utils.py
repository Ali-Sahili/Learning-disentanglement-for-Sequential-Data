import wave
import array
import numpy as np
import scipy.io.wavfile
import matplotlib.pyplot as plt




def get_signal(in_file, expected_fs=44100):
    """Load a wav file.
    If the file contains more than one channel, return a mono file by taking
    the mean of all channels.
    If the sample rate differs from the expected sample rate (default is 44100 Hz),
    raise an exception.
    Args:
        in_file: The input wav file, which should have a sample rate of `expected_fs`.
        expected_fs (int): The expected sample rate of the input wav file.
    Returns:
        The audio siganl as a 1-dim Numpy array. The values will be in the range [-1.0, 1.0]. fixme ( not yet)
    """
    fs, y = scipy.io.wavfile.read(in_file)
    num_type = y[0].dtype
    if num_type == 'int16':
        y = y*(1.0/32768)
    elif num_type == 'int32':
        y = y*(1.0/2147483648)
    elif num_type == 'float32':
        # Nothing to do
        pass
    elif num_type == 'uint8':
        raise Exception('8-bit PCM is not supported.')
    else:
        raise Exception('Unknown format.')
    if fs != expected_fs:
        raise Exception('Invalid sample rate.')
    if y.ndim == 1:
        return y
    else:
        return y.mean(axis=1)


def stft_function(input_signal, fft_size, hopsamp):
    # STFT
    window = np.hanning(fft_size)
    stft_full = np.array([np.fft.rfft(window * input_signal[i:i+fft_size])
                              for i in range(0, len(input_signal)-fft_size, hopsamp)])
    
    stft_mag = abs(stft_full)**2.0
    scale = 1.0 / np.amax(stft_mag)
    stft_mag *= scale

    return stft_mag, scale

def istft_function(stft_mag, scale, fft_size, hopsamp):

    # Undo the rescaling.
    stft_scaled = stft_mag / scale
    stft_scaled = stft_scaled**0.5

    # Inverse STFT
    window = np.hanning(fft_size)
    time_slices = stft_scaled.shape[0]
    len_samples = int(time_slices*hopsamp + fft_size)
    x = np.zeros(len_samples)
    for n,i in enumerate(range(0, len(x)-fft_size, hopsamp)):
        x[i:i+fft_size] += window*np.real(np.fft.irfft(stft_scaled[n]))

    return x


def save_audio_to_file(x, sample_rate, outfile='out.wav'):
    """Save a mono signal to a file.
    Args:
        x (1-dim Numpy array): The audio signal to save. The signal values should be in the range [-1.0, 1.0].
        sample_rate (int): The sample rate of the signal, in Hz.
        outfile: Name of the file to save.
    """
    x_max = np.max(abs(x))
    assert x_max <= 1.0, 'Input audio value is out of range. Should be in the range [-1.0, 1.0].'
    x = x*32767.0
    data = array.array('h')
    for i in range(len(x)):
        cur_samp = int(round(x[i]))
        data.append(cur_samp)
    f = wave.open(outfile, 'w')
    f.setparams((1, 2, sample_rate, 0, "NONE", "Uncompressed"))
    f.writeframes(data.tostring())
    f.close()

