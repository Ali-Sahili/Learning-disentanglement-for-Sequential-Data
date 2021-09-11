import argparse
import numpy as np
from sklearn import preprocessing

from torch.utils.data import Dataset, DataLoader

import timit_utils as tu
from utils import list_to_sparse_tensor
from features_extraction import calcfeat_delta_delta


def padding_data_lists(inputList, targetList, batchSize=1, level="phn"):
    """ 
    padding the input list to a same dimension, 
    """
    assert batchSize==1
    # dimensions of inputList: time_length*batch*39
    assert len(inputList) == len(targetList)

    nFeatures = inputList[0].shape[0]
    maxLength = 0
    for inp in inputList:
	# find the max time_length
        maxLength = max(maxLength, inp.shape[1])

    # randIxs is the shuffled index from range(0,len(inputList))
    randIxs = np.random.permutation(len(inputList))
    start, end = (0, batchSize)
    dataBatches = []

    while end <= len(inputList):
	# batchSeqLengths store the time-length of each sample in a mini-batch
        batchSeqLengths = np.zeros(batchSize)

  	# randIxs is the shuffled index of input list
        for batchI, origI in enumerate(randIxs[start:end]):
            batchSeqLengths[batchI] = inputList[origI].shape[-1]

        batchInputs = np.zeros((maxLength, batchSize, nFeatures))
        batchTargetList = []
        for batchI, origI in enumerate(randIxs[start:end]):
	    # padSecs is the length of padding
            padSecs = maxLength - inputList[origI].shape[1]
	    # numpy.pad pad the inputList[origI] with zeos at the tail
            batchInputs[:,batchI,:] = np.pad(inputList[origI].T, ((0,padSecs),(0,0)), 'constant', constant_values=0)
	    # target label
            batchTargetList.append(targetList[origI])
        dataBatches.append((batchInputs, 
                            list_to_sparse_tensor(batchTargetList, level), batchSeqLengths))
        start += batchSize
        end += batchSize
        
    return (dataBatches, maxLength) 
        
def data_load(data_path, nb_samples, batch_size, mode="train", win_len=0.025, win_step=0.001,
                    mode_feat="mfcc", feature_len=13, verbose=False):
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

    ## cleaned phonemes
    cleaned_phns = ['sil', 'aa', 'ae', 'ah', 'ao', 'aw', 'ax', 'ax-h', 'ay', 'b', 'ch', 'd', 
                    'dh', 'dx', 'eh', 'el', 'en', 'epi', 'er', 'ey', 'f', 'g', 'hh', 'ih', 
                    'ix', 'iy', 'jh', 'k', 'l', 'm', 'n', 'ng', 'ow', 'oy', 'p', 'q', 'r', 
                    's', 'sh', 't', 'th', 'uh', 'uw', 'v', 'w', 'y', 'z', 'zh']
    
    features_list = []
    labels_list = []
    count = 1
    for phn_ in original_phns:
        print("*************    " + phn_ + "    *************")
            
        s = corpus_obj.sentences_by_phone_df(phn_)
        print("number of sentences: ", len(s))
        
        for i in range(len(s)):
            print(i)
            sentence = s.sentence[i]
            wav = sentence.raw_audio
            rate = sentence.sample_rate
            words_df = sentence.words_df
            phones_df = sentence.phones_df
            if verbose: print("waveform shape: ", wav.shape)

            feat = calcfeat_delta_delta(wav,rate,win_length=win_len, win_step=win_step,
                                         mode=mode_feat,feature_len=feature_len)
            feat = preprocessing.scale(feat)
            feat = np.transpose(feat)
            if verbose: 
                print("features shape: ", feat.shape)
                print("words shape: ", words_df.shape)
                print("phones shape: ", phones_df.shape)
            
            phenome = []
            for s_, row in phones_df.iterrows():
                p_index = original_phns.index(s_)
                phenome.append(p_index)

            phenome = np.array(phenome)
            if verbose: 
                #print(phenome.shape)
                print()
            
            if count > nb_samples:
                break
            
            features_list.append(feat)
            labels_list.append(phenome)
            
            count += 1
            
        print("*************************************")

    assert len(features_list) == len(labels_list)
    if verbose: print("number of samples: ", len(features_list))
     
    dataBatches, maxLength = padding_data_lists(features_list, labels_list)
    batchInputs, Labels, batchSeqLengths = dataBatches[0][0],dataBatches[0][1], dataBatches[0][2]
        
    return dataBatches, maxLength

class Timit(Dataset):
    def __init__(self, data_path, nb_samples=100, win_len=0.025, win_step=0.001, feature_len=26, 
                        mode="train", batch_size=2, mode_feat="mfcc", verbose=False):
        
        dataBatches, maxLength = data_load(data_path, nb_samples, batch_size, mode, win_len, 
                                                      win_step, mode_feat, feature_len, verbose)
        self.data, self.labels, _ = zip(*dataBatches)

    def __getitem__(self, item):
        return self.data[item].squeeze(1)

    def __len__(self):
        return len(self.data)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(prog='timit_preprocess',description="Visualize timit data ")
    parser.add_argument("--data_path",type=str,default='../../TIMIT_dataset/org_dataset/timit/data',help="Directo of Timit data")

    parser.add_argument("--mode", help="Mode",choices=['train','test'],type=str,default='train')
    parser.add_argument("--mode_feat", help="Mode of feature extraction", 
                                             choices=['mfcc','fbank'], type=str, default='mfcc')
    parser.add_argument('--batch_size', type=int, default=2, help='Features length')    
    parser.add_argument("--verbose", default=False, help="set this flag to print dimensions.")

    parser.add_argument('--nb_samples', type=int, default=100, help='nb of samples')        
    parser.add_argument('--feature_len', type=int, default=13, help='Features length')
    parser.add_argument("--win_len", type=float,default=0.02,help="the window length of feature")
    parser.add_argument("--win_step",type=float,default=0.01,help="window step length of feats")

    args = parser.parse_args()
    
    timit = Timit(args.data_path, args.nb_samples, win_len=args.win_len, win_step=args.win_step, 
                   feature_len=args.feature_len, verbose=args.verbose, mode=args.mode, 
                   batch_size=args.batch_size, mode_feat=args.mode_feat)
    data_loader = DataLoader(timit, batch_size=args.batch_size, shuffle=True, num_workers=4,
                                     persistent_workers = True)
    
    print("we have ", len(data_loader.dataset), " samples.")    
    sample = next(iter(data_loader))
    print("dimension of each sample: ", sample.shape)

