import os
import numpy as np
import soundfile as sf
import librosa
import random
import torch
from torch.utils import data

class SpeechSequencesFull(data.Dataset):
    """
    Customize a dataset of speech sequences for Pytorch
    at least the three following functions should be defined.
    """
    def __init__(self, file_list, sequence_len, shuffle=True,
                  fs=16000, nfft=512, hop=5, wlen=25):

        super().__init__()

        # STFT parameters
        self.fs = fs
        self.nfft = nfft
        self.hop = hop
        self.wlen = wlen
        
        # data parameters
        self.file_list = file_list
        self.sequence_len = sequence_len
        self.shuffle = shuffle

        self.compute_len()

    def compute_len(self):

        self.valid_seq_list = []

        for wavfile in self.file_list:

            x, fs_x = sf.read(wavfile)
            if self.fs != fs_x:
                raise ValueError('Unexpected sampling rate')
            
            # remove beginning and ending silence
            path, file_name = os.path.split(wavfile)
            path, speaker = os.path.split(path)
            path, dialect = os.path.split(path)
            path, set_type = os.path.split(path)
            with open(os.path.join(path, set_type, dialect, speaker, file_name[:-4] + '.PHN'), 'r') as f:
                first_line = f.readline() # Read the first line
                for last_line in f: # Loop through the whole file reading it all
                    pass
            if not('#' in first_line) or not('#' in last_line):
                raise NameError('The first of last lines of the .phn file should contain #')
            ind_beg = int(first_line.split(' ')[1])
            ind_end = int(last_line.split(' ')[0])

            # Check valid wav files
            seq_length = (self.sequence_len - 1) * self.hop
            file_length = ind_end - ind_beg 
            n_seq = (1 + int(file_length / self.hop)) // self.sequence_len
            for i in range(n_seq):
                seq_start = i * seq_length + ind_beg
                seq_end = (i + 1) * seq_length + ind_beg
                seq_info = (wavfile, seq_start, seq_end)
                self.valid_seq_list.append(seq_info)

        if self.shuffle:
            random.shuffle(self.valid_seq_list)


    def __len__(self):
        """
        arguments should not be modified
        Return the total number of samples
        """
        return len(self.valid_seq_list)


    def __getitem__(self, index):
        """
        input arguments should not be modified
        torch data loader will use this function to read ONE sample of data from a list that can be indexed by
        parameter 'index'
        """
        
        # Read wav files
        wavfile, seq_start, seq_end = self.valid_seq_list[index]
        #print(seq_start, seq_end)
        x, fs_x = sf.read(wavfile)

        # Sequence tailor
        x = x[int(seq_start):int(seq_end)]

        # Normalize sequence
        x = x/np.max(np.abs(x))

        # STFT transformation
        audio_spec = torch.stft(torch.from_numpy(x), n_fft=self.nfft, hop_length=self.hop, 
                                win_length=self.wlen, #window=25, 
                                center=True, pad_mode='reflect', normalized=False, onesided=True)

        # Square of magnitude
        sample = (audio_spec[:,:,0]**2 + audio_spec[:,:,1]**2).float()

        return sample


                
class SpeechSequencesRandom(data.Dataset):
    """
    Customize a dataset of speech sequences for Pytorch
    at least the three following functions should be defined.
    
    This is a quick speech sequence data loader which allow multiple workers
    """
    def __init__(self, file_list, sequence_len, shuffle=True,
                  fs=16000, nfft=512, hop=5, wlen=25):

        super().__init__()

        # STFT parameters
        self.fs = fs
        self.nfft = nfft
        self.hop = hop
        self.wlen = wlen
        
        # data parameters
        self.file_list = file_list
        self.sequence_len = sequence_len
        self.shuffle = shuffle

        self.compute_len()


    def compute_len(self):

        self.valid_file_list = []

        for wavfile in self.file_list:

            x, fs_x = sf.read(wavfile)
            if self.fs != fs_x:
                raise ValueError('Unexpected sampling rate')

            # Check valid wav files
            seq_length = (self.sequence_len - 1) * self.hop
            if len(x) >= seq_length:
                self.valid_file_list.append(wavfile)

        if self.shuffle:
            random.shuffle(self.valid_file_list)


    def __len__(self):
        """
        arguments should not be modified
        Return the total number of samples
        """
        return len(self.valid_file_list)


    def __getitem__(self, index):
        """
        input arguments should not be modified
        torch data loader will use this function to read ONE sample of data from a list that can be indexed by
        parameter 'index'
        """
        
        # Read wav files
        wavfile = self.valid_file_list[index]
        x, fs_x = sf.read(wavfile)

        # Silence clipping
        path, file_name = os.path.split(wavfile)
        path, speaker = os.path.split(path)
        path, dialect = os.path.split(path)
        path, set_type = os.path.split(path)
        with open(os.path.join(path, set_type, dialect, speaker, file_name[:-4] + '.PHN'), 'r') as f:
            first_line = f.readline() # Read the first line
            for last_line in f: # Loop through the whole file reading it all
                pass
        if not('#' in first_line) or not('#' in last_line):
            raise NameError('The first of last lines of the .phn file should contain #')
        ind_beg = int(first_line.split(' ')[1])
        ind_end = int(last_line.split(' ')[0])
        x = x[ind_beg:ind_end]

        # Sequence tailor
        file_length = len(x)
        seq_length = (self.sequence_len - 1) * self.hop # sequence length in time domain
        start = np.random.randint(0, file_length - seq_length)
        end = start + seq_length
        x = x[start:end]

        # Normalize sequence
        x = x/np.max(np.abs(x))

        # STFT transformation
        audio_spec = torch.stft(torch.from_numpy(x), n_fft=self.nfft, hop_length=self.hop, 
                                win_length=self.wlen, #window=self.win, 
                                center=True, pad_mode='reflect', normalized=False, onesided=True)

        # Square of magnitude
        sample = (audio_spec[:,:,0]**2 + audio_spec[:,:,1]**2).float()

        return sample


