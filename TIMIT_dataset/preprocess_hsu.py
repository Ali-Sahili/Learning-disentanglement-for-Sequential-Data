import argparse
import numpy as np

from torch.utils.data import Dataset, DataLoader

import timit_utils as tu
import timit_utils.audio_utils as au

import matplotlib.pyplot as plt
import scipy

# https://superkogito.github.io/blog/SignalFraming.html
def framing(sig, fs=16000, win_len=0.2, win_hop=0.001):
     """
     transform a signal into a series of overlapping frames.

     Args:
         sig            (array) : a mono audio signal (Nx1) from which to compute features.
         fs               (int) : the sampling frequency of the signal we are working with.
                                  Default is 16000.
         win_len        (float) : window length in sec.
                                  Default is 0.2. --> for frames of 200 ms
         win_hop        (float) : step between successive windows in sec.
                                  Default is 0.01.

     Returns:
         array of frames.
         frame length.
     """
     # compute frame length and frame step (convert from seconds to samples)
     frame_length = win_len * fs
     frame_step = win_hop * fs
     signal_length = len(sig)
     frames_overlap = frame_length - frame_step

     # Make sure that we have at least 1 frame+
     num_frames = np.abs(signal_length - frames_overlap)// np.abs(frame_length - frames_overlap)
     rest_samples = np.abs(signal_length-frames_overlap) % np.abs(frame_length - frames_overlap)

     # Pad Signal to make sure that all frames have equal number of samples
     # without truncating any samples from the original signal
     if rest_samples != 0:
         pad_signal_length = int(frame_step - rest_samples)
         z = np.zeros((pad_signal_length))
         pad_signal = np.append(sig, z)
         num_frames += 1
     else:
         pad_signal = sig

     # make sure to use integers as indices
     frame_length = int(frame_length)
     frame_step = int(frame_step)
     num_frames = int(num_frames)

     # compute indices
     idx1 = np.tile(np.arange(0, frame_length), (num_frames, 1))
     idx2 = np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
     indices = idx1 + idx2
     frames = pad_signal[indices.astype(np.int32, copy=False)]
     return frames, frame_length


def plot_wav(frame, rate, color='r'):
    times = np.arange(len(frame))/float(rate) * 1000
    plt.figure(figsize=(30, 4))
    plt.fill_between(times, frame, color=color)  
    plt.xlim(times[0], times[-1])
    plt.xlabel('time (ms)')
    plt.ylabel('amplitude')

def plot_freq(frame_fft, rate, color="r"):
    N = frame_fft.shape[0]
    secs = N / float(rate)
    Ts = 1.0/rate   # sampling interval in time
    t = scipy.arange(0, secs, Ts)       # time vector as scipy arange field / numpy.ndarray
    freqs = scipy.fftpack.fftfreq(N, t[1]-t[0])
    plt.figure(figsize=(30, 4))
    plt.plot(freqs, frame_fft, color=color)

    freqs_side = freqs[range(N//2)]
    frame_fft_side = frame_fft[range(N//2)]
    plt.figure(figsize=(30, 4))
    plt.plot(freqs_side, abs(frame_fft_side), color=color)

    ims = 20.*np.log10(np.abs(frame_fft)/10e-6)
    plt.figure(figsize=(30, 4))
    plt.plot(ims, color=color)


def Extract_Features(frame, rate, mode, win_len, win_step, feature_len, nfft=512, verbose=False):
    from python_speech_features import mfcc, fbank
    
    if mode == "fbank":
        mel_filterbank, mel_en = fbank(frame,samplerate=rate,winlen=win_len, winstep=win_step, 
                                       nfilt=feature_len, nfft=nfft, lowfreq=0, highfreq=None, 
                                       preemph=0.97)
        mel_energies = mel_en.reshape((len(mel_en),1))
        mel_filterbank_log = np.log(mel_filterbank + 1)
        if verbose:
            print("FBank: ", mel_filterbank.shape)
            print("FBank energy: ", mel_en.shape)
        return mel_filterbank
    elif mode == "mfcc":
        mfcc_features = mfcc(frame, samplerate=rate, winlen=win_len, winstep=win_step, 
                             nfilt=feature_len, nfft=nfft, lowfreq=0, highfreq=None, 
                             preemph=0.97)
        if verbose: print("MFCC: ", mfcc_features.shape)
        return mfcc_features
    else:
        raise NotImplemented
 


def load_padded_features(data_path, nb_samples, win_len=0.025, win_step=0.001, feature_len=26, 
                           mode="train", verbose=False):
    corpus = tu.Corpus(data_path)
    
    if mode == "train":
        corpus_obj = corpus.train 
    elif mode == "test":
        corpus_obj = corpus.test
        
    ## original phonemes                    
    original_phns = ['aa', 'ae', 'ah', 'ao', 'aw', 'ax', 'ax-h', 'axr', 'ay', 'b', 'bcl', 'ch', 
                     'd', 'dcl', 'dh', 'dx', 'eh', 'el', 'em', 'en', 'eng', 'epi', 'er', 'ey', 
                     'f', 'g', 'gcl', 'hh', 'hv', 'ih', 'ix', 'iy', 'jh', 'k', 'kcl', 'l', 
                     'm', 'n', 'ng', 'nx', 'ow', 'oy', 'p', 'pau', 'pcl', 'q', 'r', 's', 'sh', 
                     't', 'tcl', 'th', 'uh', 'uw', 'ux', 'v', 'w', 'y', 'z', 'zh']

    frames_list = []
    rate_list = []
    count = 1
    max_len = -1
    for phn_ in original_phns:
        print("*************    " + phn_ + "    *************")   
        s = corpus_obj.sentences_by_phone_df(phn_)
        print("number of sentences: ", len(s))
        
        for i in range(len(s)):
            print(i)
            sentence = s.sentence[i]
            wav = sentence.raw_audio
            rate = sentence.sample_rate
            frames_tmp, frame_length = framing(np.array(wav).reshape(-1,1))
            frames_tmp = frames_tmp.reshape(-1, frame_length)
            #print(frames_tmp.shape)

            if count > nb_samples:
                break
                
            frames_list.append(frames_tmp)
            rate_list.append(rate)

            count += 1

        if count > nb_samples:
            break
        print("*************************************")
        
    frames = np.concatenate(frames_list, axis=0)
    sample_rates = np.array(rate_list)
    print(frames.shape, sample_rates.shape)
    
    if False:
      plot_wav(frames[10], rate)
      FFT = abs(scipy.fft(frames[10]))
      plot_freq(FFT, rate)
      plt.show()

    features_list = []
    
    for i in range(frames.shape[0]):
        print(i)
        #feat = Extract_Features(frames[i], rate, "fbank", win_len, win_step, feature_len)
        feat = Extract_Features(frames[i], rate, "mfcc", win_len, win_step, feature_len)
        # Combination of several types of features
        #feat = au.audio_features(frames[i], rate, win_len, win_step, feature_len) 
        #print("Features: ", feat.shape)

        features_list.append(feat)

    return features_list


class Timit(Dataset):
    def __init__(self, data_path, nb_samples=100, win_len=0.025, win_step=0.001, feature_len=26, 
                           mode="train", verbose=False):
        
        features_list = load_padded_features(data_path, nb_samples, win_len, win_step, 
                                               feature_len, mode, verbose)
        self.data = features_list
        self.label = None
        
    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(prog='timit_preprocess',description="Visualize timit data ")
    parser.add_argument("--data_path",type=str,default='../../TIMIT_dataset/org_dataset/timit/data',help="Directo of Timit data")

    parser.add_argument("--mode", help="Mode",choices=['train','test'],type=str,default='train')
    parser.add_argument('--batch_size', type=int, default=2, help='Features length')    
    parser.add_argument("--verbose", default=False, help="set this flag to print dimensions.")
    
    parser.add_argument('--nb_samples', type=int, default=10, help='number od samples')        
    parser.add_argument('--feature_len', type=int, default=26, help='Features length')    
    parser.add_argument("--win_len",type=float,default=0.025,help="the window length of feature")
    parser.add_argument("--win_step",type=float,default=0.001,help="window step length of feats")

    args = parser.parse_args()
    
    timit = Timit(args.data_path, args.nb_samples, win_len=args.win_len, win_step=args.win_step, 
                        feature_len=args.feature_len, verbose=args.verbose, mode=args.mode)
    data_loader = DataLoader(timit, batch_size=args.batch_size, shuffle=True, num_workers=4,
                                     persistent_workers = True)
    
    print("we have ", len(data_loader.dataset), " samples.")    
    sample = next(iter(data_loader))
    print("dimension of each sample: ", sample.shape)
        
