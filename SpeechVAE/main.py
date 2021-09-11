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
    model = train(args, tr_iterator, dt_iterator, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='VAE-Variants for Learning Disentanglement.')
    
    parser.add_argument('--method', default='SpeechVAE', type=str, help='FHVAE/DSVAE/SpeechVAE/Dynamical_VAEs.')

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

    # Speech-VAE Parameters
    parser.add_argument('--model', default='VAE', type=str, help='VAE/CVEA/LSTMVAE...')
    parser.add_argument('--optimizer', default='Adam', type=str, help='optimizer.')
    parser.add_argument('--BCE', default=False, type=bool, help='choose loss')
    parser.add_argument("--alpha",type=float,default=1.,help="weight for KLD")
    parser.add_argument("--beta",type=float,default=4.,help="weight for KLD")
    parser.add_argument("--gamma",type=float,default=1.,help="weight for Reconstruction loss")
    parser.add_argument("--clip",type=float,default=4.,help="gradient clipping value")
    parser.add_argument('--C_max', default=25.,type=float, help='capacity of bottleneck channel')
    parser.add_argument('--C_stop_iter', default=1e3,type=float,help='stop increasing capacity')
    parser.add_argument('--anneal_steps', default=200, type=int, help='gamma of beta_TCVAE')
    parser.add_argument('--penalty', default=False, type=bool, help='choose loss')
    parser.add_argument('--f_penalty', default=0.1,type=float,help='factor penalty')
    args = parser.parse_args()

    main(args)
