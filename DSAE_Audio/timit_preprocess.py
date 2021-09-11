
import numpy as np
import scipy.io.wavfile
import matplotlib.pyplot as plt

from utils import get_signal, stft_function, istft_function



def run_demo(data_dir, sample_rate, fft_size,hopsamp,seq_len=20,nb_samples=10):

    filenames = []
    with open(data_dir+'wav.txt') as f:
        lines = f.readlines()
        print(len(lines))
        
        
        for line in lines:
            line = line.strip()
            if line[-3:] == "WAV":
                filenames.append(line.split(' ')[-1])

        print(len(filenames))

    print()


    # Initialize
    stft_data = []
    len_data = []
    scale_data = []
    count = 1
    for name in filenames:
        print(count)
        input_signal = get_signal(name, expected_fs=sample_rate)

        stft_mag, scale = stft_function(input_signal, fft_size, hopsamp)

        len_data.append(stft_mag.shape[0])
        scale_data.append(scale)
        stft_data.append(stft_mag)

        count += 1

        if count == nb_samples:
            break

    len_data = np.array(len_data)
    scale_data = np.array(scale_data)

    max_length = np.max(len_data)
    while max_length % seq_len > 0:
        max_length -= 1
    print("max length: ", max_length)

    data = []
    for i in range(len_data.shape[0]):

        signal = stft_data[i]
        pad_length = max_length - len_data[i]

        if pad_length < 0:
            pad_signal = np.delete(signal,range(len_data[i]-abs(pad_length),len_data[i]),axis=0) 
        else:
            z = np.zeros((pad_length,257))
            pad_signal = np.concatenate([signal, z], axis=0)

        pad_signal = np.expand_dims(pad_signal, axis=0)
        data.append(pad_signal)


    stft_data = np.concatenate(data, axis=0)
    print("data dimensions: ", stft_data.shape)

    stft_data = np.split(stft_data, max_length/seq_len, axis=1)
    print("data dimensions: ", len(stft_data), stft_data[0].shape)

    stft_data = np.concatenate(stft_data, axis=0)
    print("data dimensions: ", stft_data.shape)

    np.save('length.npy', len_data)
    np.save('scale.npy', scale_data)
    np.save('features.npy', stft_data)


sample_rate = 16000
fft_size = 512
hopsamp = fft_size // 8

seq_len = 20
nb_samples = 500

data_dir = "data/" #

run_demo(data_dir, sample_rate, fft_size, hopsamp, seq_len, nb_samples)
