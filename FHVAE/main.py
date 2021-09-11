import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader

from dataset import load_data

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


def main(args):
    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    #data_loader = return_data(args)
    device = 'cuda' if torch.cuda.is_available() and args.use_cuda else 'cpu'
    print(device)
    
    #tr_dset, dt_dset, tr_iterator, dt_iterator = load_data("preprocessed_dataset") # fbank
    tr_dset, dt_dset, tr_iterator, dt_iterator = load_data("preprocessed_dataset_stft")  # stft
    
    print()
    print("Data is generated ...")
    print("size of training set: ", len(tr_dset.seqlist))
    print()
    
    from train import train
    model = train(args, tr_iterator, dt_iterator, tr_dset, dt_dset, device)
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='VAE-Variants for Learning Disentanglement.')
    
    parser.add_argument('--seed', default=1, type=int, help='random seed')
    parser.add_argument('--use_cuda', default=True, type=bool, help='enable cuda')
    parser.add_argument('--save_plot', default=True, type=bool, help='save losses plots')
    
    # I/O Paths Parameters
    parser.add_argument('--save_output', default=True, type=bool, help='save images and gif')
    parser.add_argument('--output_dir', default='outputs', type=str, help='output directory')
    
    parser.add_argument('--ckpt_dir', default='checkpoints',type=str, help='ckpoint directory')
    parser.add_argument('--ckpt_iter', default=1000, type=int, help='load specific checkpoint.')

    # Dataset setting Parameters
    parser.add_argument("--data_path",type=str,default='../../TIMIT_dataset/org_dataset/timit/data',help="Directo of Timit data")
    parser.add_argument('--batch_size', default=256, type=int, help='batch size')
    parser.add_argument('--num_workers', default=2, type=int, help='dataloader num_workers')

    # training Settings
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--num_epochs', default=10, type=int, help='number of epochs')
    parser.add_argument('--beta1', default=0.9, type=float, help='Adam optimizer beta1')
    parser.add_argument('--beta2', default=0.999, type=float, help='Adam optimizer beta2')
    parser.add_argument('--display_step', default=1, type=int,help='print res every n iters.')
    parser.add_argument('--save_step', default=1, type=int, help='saving every n iters.')
    
    # FHVAE Parameters
    parser.add_argument("--data_dir", type=str, default=None, help="Location of the data")
    parser.add_argument("--z1-hus", default=[128,128], nargs=2, help="List hidden units of z1")
    parser.add_argument("--z2-hus", default=[128,128], nargs=2, help="List hidden units of z2")
    parser.add_argument("--z1-dim", type=int, default=16, help="Dimension of the z1 layer")
    parser.add_argument("--z2-dim", type=int, default=16, help="Dimension of the z2 layer")
    parser.add_argument("--x-hus", default=[128, 128], nargs=2, help="List of hidden units ")
    parser.add_argument("--alpha_dis",type=float,default=10.,help="Discrimin objective weight")
    parser.add_argument("--sample_rate",type=int,default=None,help="Rate to resample audio")
    parser.add_argument("--win_size",type=float,default=0.025,help="Window size for spectrogram\
                                                                        in seconds")
    parser.add_argument("--hop_size",type=float,default=0.010,help="Window stride for \
                                                                      spectrogram in seconds")
    parser.add_argument("--mels", type=int, default=80, help="Number of filter banks")
    parser.add_argument("--min-len",type=int,default=None, help="Minimum segment length.")
    parser.add_argument("--mvn-path",type=str,default=None,help="Path to a precomputed mean and\
                                                                 variance normalization file")
    parser.add_argument("--seg-len", type=int, default=20, help="Segment length to use")
    parser.add_argument("--seg-shift",type=int,default=8, help="Segment shift if rand_seg is\
             False, otherwise floor(seq_len/seg_shift) segments per sequence will be extracted")

    args = parser.parse_args()

    main(args)
