import os
import torch
import numpy as np
from plot import plot
from FHVAE.model import FHVAE


def save_checkpoints(ckpt_dir, epoch, model, optimizer):
    """Saves checkpoint files"""

    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    
    print("Saving...")
    checkpoint = { "epoch": epoch,
                   "optimizer": optimizer.state_dict(),
                   "state_dict": model.state_dict()
                 }

    f_path = ckpt_dir + "/FHVAE_Epoch_" + str(epoch) + ".pth"
    torch.save(checkpoint, f_path)
    

# alpha/discriminative weight of 10 was found to produce best results
def loss_function(lower_bound, log_qy, alpha=10.0):
    """Discriminative segment variational lower bound
    Returns:
        Segment variational lower bound plus the (weighted) discriminative objective.
    """
    return -1 * torch.mean(lower_bound + alpha * log_qy)


def train(args, tr_iterator, dt_iterator, tr_dset, dt_dset, device):
    
    tr_nseqs = len(tr_dset.seqlist)
    tr_shape = tr_dset.get_shape()
    
    dt_nseqs = len(dt_dset.seqlist)
    dt_shape = dt_dset.get_shape()
    
    input_size = np.prod(tr_shape)
    
    # load model
    model = FHVAE(input_size, args.z1_hus, args.z2_hus, args.z1_dim, args.z2_dim, args.x_hus,
                                                     device=device)

    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr,betas=(args.beta1,args.beta2))

    model.to(device)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    ckpt_file = args.ckpt_dir + "/FHVAE_Epoch_" + str(args.ckpt_iter) + ".pth"
    if os.path.exists(ckpt_file):
        print("Loading checkpoint...")
        checkpoint = torch.load(ckpt_file)
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])

    model.double()
    print("Start Training...")
    for epoch in range(args.num_epochs):
        # training
        model.train()
        train_loss = 0.0
        for i, (x_val, y_val, n_val) in enumerate(tr_iterator()):
            features = torch.from_numpy(np.stack(x_val, axis=0)).to(device)
            idxs = torch.tensor([idx for idx in y_val]).to(device)
            nsegs = torch.from_numpy(np.array(n_val))
            # print(features.shape, idxs.shape, nsegs.shape)
            
            lower_bound, discrim_loss, log_px_z, neg_kld_z1, neg_kld_z2, log_pmu2, z1_sample, z2_sample, x_sample = model(features, idxs, tr_nseqs, nsegs)

            loss = loss_function(lower_bound.cpu(), discrim_loss.cpu(), args.alpha_dis)

            optimizer.zero_grad()
            loss.backward()
            train_loss += loss.item()
            optimizer.step()

            if i % args.display_step == 0:
                cur_loss = loss.item() / len(features)
                print(f"====> Train set | Epoch: {epoch}/{args.num_epochs} | batch: {i} | Loss: {cur_loss:.6f}")
            
        train_loss /= tr_nseqs
        print(f"====> Train set average loss: {train_loss:.4f}")
        print()
        
        # plotting
        plot(x_val, args.output_dir, img="image", epoch=epoch)
        plot(x_sample.detach().cpu().numpy(), args.output_dir, img="result", epoch=epoch)
        
        if epoch % args.save_step:
            save_checkpoints(args.ckpt_dir, epoch, model, optimizer)
        
        # eval
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_val, y_val, n_val in dt_iterator():
                feature = torch.from_numpy(np.stack(x_val, axis=0)).to(device)
                idxs = torch.tensor([idx for idx in y_val]).to(device)
                nsegs = torch.from_numpy(np.array(n_val))
                
                lower_bound, discrim_loss, log_px_z, neg_kld_z1, neg_kld_z2, log_pmu2, z1_sample, z2_sample, x_sample = model(feature, idxs, dt_nseqs, nsegs)
                cur_loss = loss_function(lower_bound.cpu(), discrim_loss.cpu(), 
                                          args.alpha_dis).item()
                val_loss += cur_loss
                cur_loss = cur_loss / len(feature)

                #print(f"====> Validation Epoch: {epoch} Loss: {cur_loss:.6f}")

        val_loss /= dt_nseqs
        print(f"====> Validation set | Epoch: {epoch} | Loss: {val_loss:.4f}")
        
    return model

if __name__ == '__main__':

    parser = argparse.ArgumentParser(prog='training',description="train the model ")

    parser.add_argument("--data_dir", type=str, default=None, help="Location of the data")
    parser.add_argument("--batch_size", type=int, default=2048, help="Batch size")
    parser.add_argument("--z1-hus", default=[128,128], nargs=2, help="List hidden units of z1")
    parser.add_argument("--z2-hus", default=[128,128], nargs=2, help="List hidden units of z2")
    parser.add_argument("--z1-dim", type=int, default=16, help="Dimension of the z1 layer")
    parser.add_argument("--z2-dim", type=int, default=16, help="Dimension of the z2 layer")
    parser.add_argument("--x-hus", default=[128, 128], nargs=2, help="List of hidden units per\
                                                    layer for the pre-stochastic layer decoder")
    parser.add_argument("--use_cuda", type=bool, default=True, help="to be set for gpu use.")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--beta1", type=float, default=0.95, help="Beta1 for the Adam optimizer")
    parser.add_argument("--beta2", type=float, default=0.999,help="Beta2 for the Adam optimizer")

    parser.add_argument("--alpha_dis",type=float,default=10.,help="Discrimin objective weight")
    parser.add_argument("--num_epochs", type=int,default=100, help="Number of training epochs")
    parser.add_argument("--display_step", type=int,default=10, help="display losses")

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

