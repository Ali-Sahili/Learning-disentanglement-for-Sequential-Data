import argparse
import numpy as np

from torch.utils.data import Dataset, DataLoader

import timit_utils as tu
import timit_utils.audio_utils as au


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

    wav_list = []
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
            if verbose: print("waveform shape: ", wav.shape)

            if count > nb_samples:
                break

            wav_len = wav.shape[0]
            if max_len < wav_len:
                max_len = wav_len

            wav_list.append(wav)
            rate_list.append(rate)

            count += 1
            
        print("*************************************")
    
    wav_list_pad = []
    features_list = []
    for i,(wav,rate) in enumerate(zip(wav_list,rate_list)):
        print(i)
        
        wav_len = wav.shape[0]
        pad_len = max_len - wav_len
        
        pad_left = pad_len//2
        if (pad_len/2) > (pad_len//2) :
          pad_right = pad_len//2 + 1
        else:
          pad_right = pad_len//2
        
        wav_pad = au.audio_zero_padded(pad_left, wav, pad_right)    
        wav_list_pad.append(wav_pad)

        feat = au.audio_features(wav_pad, rate, win_len, win_step, feature_len)
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
    
    parser.add_argument('--nb_samples', type=int, default=100, help='number od samples')        
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
        
