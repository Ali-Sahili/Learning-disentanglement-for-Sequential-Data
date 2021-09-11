import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
from hessian_penalty import hessian_penalty
from plot import plot, plot_latent_space
from SpeechVAE.losses import TCVAE_loss_function

def train(args, tr_iterator, dt_iterator, device):

    if args.model == "Beta_VAE" or args.model == "Annealed_VAE" or args.model == "Beta_TCVAE":
        from SpeechVAE.vae import VAE
        Z_DIM = 32
        model = VAE(input_size=19*80,h_dim=[4096,2048,1024,512],z_dim=Z_DIM,dropout=0.,relu=0.2)
    elif args.model == "Beta_CVAE" or args.model == "Annealed_CVAE" or args.model == "conv_Beta_TCVAE":
        from SpeechVAE.cvae import VAE
        Z_DIM = 32
        model = VAE(z_dim=Z_DIM, log_var_=None, dropout=0, relu=0.2, n_filters=80)
    elif args.model == "Beta_GCN_VAE" or args.model == "Annealed_GCN_VAE":
        from SpeechVAE.gcnvae import VAE
        Z_DIM = 32
        model = VAE(z_dim=Z_DIM, log_var_=None, dropout=0, relu=0.2, n_filters=80)
    else:
        raise NotImplemented

    # Optimizers
    if args.optimizer == "Adam":
        optimizer = torch.optim.Adam(model.parameters(),
                               lr=args.lr,
                               betas=(args.beta1,args.beta2),
                               eps=1e-8,
                               weight_decay=1e-4)
    else:
        optimizer = torch.optim.SGD(model.parameters(),
                              lr=args.lr,
                              momentum=args.beta1,
                              nesterov=True,
                              weight_decay=1e-4)

    model.to(device)
    print(model)

    #scheduler = StepLR(optimizer, step_size=100, gamma=0.5)
    add_penalty = args.penalty
    CLIP = args.clip # 0.25

    for epoch in range(args.num_epochs):
        model.train()
        loss_epoch = 0
        recon_loss_epoch = 0
        kl_loss_epoch = 0
        for i, (x_val, _, _) in enumerate(tr_iterator()):

            data_i = torch.from_numpy(np.stack(x_val, axis=0)).float().to(device)[:,:19]
            
            out, mu, log_var, z = model(data_i)

            if args.model == "Beta_VAE" or args.model == "Beta_CVAE" or args.model == "Beta_LSTM_VAE" or args.model == "Beta_GCN_VAE":
                # Compute reconstruction loss and kL divergence
                if args.BCE:
                    reconst_loss = F.binary_cross_entropy(F.sigmoid(out), F.sigmoid(data_i), 
                                                                         size_average=False)
                else:
                    reconst_loss = F.mse_loss(out, data_i, size_average=False)
                kl_divergence = torch.sum(0.5 * (mu ** 2 + torch.exp(log_var) - log_var - 1))
                
                reconst_loss /= data_i.shape[0] * data_i.shape[1]
                kl_divergence /= data_i.shape[0] * data_i.shape[1]
                total_loss = reconst_loss + args.beta * kl_divergence
                
            elif args.model == "Annealed_VAE" or args.model == "Annealed_CVAE" or args.model == "Annealed_LSTM_VAE" or args.model == "Annealed_GCN_VAE":
                # Compute reconstruction loss and kL divergence
                if args.BCE:
                    reconst_loss = F.binary_cross_entropy(F.sigmoid(out), F.sigmoid(data_i), 
                                                                         size_average=False)
                else:
                    reconst_loss = F.mse_loss(out, data_i, size_average=False)
                kl_divergence = torch.sum(0.5 * (mu ** 2 + torch.exp(log_var) - log_var - 1))
                
                C_max = Variable(torch.FloatTensor([args.C_max]).to(device))
                C = torch.clamp(C_max/args.C_stop_iter*i, 0, C_max.data[0])
                
                reconst_loss /= data_i.shape[0] * data_i.shape[1]
                kl_divergence /= data_i.shape[0] * data_i.shape[1]
                total_loss = reconst_loss + args.gamma*(kl_divergence-C).abs()
                
            elif args.model == "Beta_TCVAE" or args.model == "conv_Beta_TCVAE":
                dataset_size = 1580 * 256
                total_loss, reconst_loss, kl_divergence, tc_loss, mi_loss = TCVAE_loss_function(data_i, out, mu, log_var, z, dataset_size, batch_iter=i, 
 anneal_steps=args.anneal_steps, alpha=args.alpha, beta=args.beta, gamma=args.gamma, train=True)
                
                total_loss, reconst_loss, kl_divergence, tc_loss, mi_loss = TCVAE_loss_function(data_i.contiguous().view(-1, 80), out.contiguous().view(-1, 80), (z_mean-z_mean_p).contiguous().view(-1, Z_DIM), (z_logvar-z_logvar_p).contiguous().view(-1, Z_DIM), z.contiguous().view(-1, Z_DIM), dataset_size, batch_iter=i, anneal_steps=args.anneal_steps, alpha=args.alpha, beta=args.beta, gamma=args.gamma, train=True)
            
            
            if add_penalty: 
                penalty = hessian_penalty(G=model, z=data_i)/(256*19)
                total_loss += penalty*args.f_penalty
             
            # Back propagation + Optimize
            optimizer.zero_grad()
            total_loss.backward()
            #torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
            optimizer.step()
            #scheduler.step()
            
            loss = total_loss.item()
            #recon_loss_epoch += reconst_loss.item()
            #kl_loss_epoch += kl_divergence.item()

            if i % args.display_step == 0:
                if add_penalty: 
                    print(f"Epoch: {epoch}/{args.num_epochs} | Batch: {i} | Recons: {reconst_loss:.4f} | KLD: {kl_divergence:.4f} | Penalty: {penalty:.4f}")
                else: 
                    print(f"Epoch: {epoch}/{args.num_epochs} | Batch: {i} | Recons: {reconst_loss:.4f} | KLD: {kl_divergence:.4f} | Loss: {loss:.4f}")

        print(f"Epoch: {epoch}/{args.num_epochs} | Recons: {recon_loss_epoch/i:.4f} | KLD: {kl_loss_epoch/i:.4f}")
        
        # plotting
        plot(data_i.cpu().numpy(), args.output_dir, img="image", epoch=epoch)
        plot(out.detach().cpu().numpy(), args.output_dir, img="result", epoch=epoch) 
        plot_latent_space(z.detach().cpu().numpy(), args.output_dir, epoch)
        
        if args.save_step:
            print("Saving...")

        model.eval()
        with torch.no_grad():
            for i, (x_val, _, _) in enumerate(dt_iterator()):

                data_i = torch.from_numpy(np.stack(x_val, axis=0)).float().to(device)[:,:19]
                
                out, mu, log_var, z = model(data_i)

                if args.model == "Beta_VAE" or args.model == "Beta_CVAE" or args.model == "Beta_LSTM_VAE" or args.model == "Beta_GCN_VAE":
                    # Compute reconstruction loss and kL divergence
                    if args.BCE:
                        reconst_loss = F.binary_cross_entropy(F.sigmoid(out), F.sigmoid(data_i), 
                                                                         size_average=False)
                    else:
                        reconst_loss = F.mse_loss(out, data_i, size_average=False)
                    kl_divergence = torch.sum(0.5 * (mu ** 2 + torch.exp(log_var) - log_var - 1))
                    
                    reconst_loss /= data_i.shape[0] * data_i.shape[1]
                    kl_divergence /= data_i.shape[0] * data_i.shape[1]
                    total_loss = reconst_loss + args.beta * kl_divergence
                
                elif args.model == "Annealed_VAE" or args.model == "Annealed_CVAE" or args.model == "Annealed_LSTM_VAE" or args.model == "Annealed_GCN_VAE":
                    # Compute reconstruction loss and kL divergence
                    if args.BCE:
                        reconst_loss = F.binary_cross_entropy(F.sigmoid(out), F.sigmoid(data_i), 
                                                                         size_average=False)
                    else:
                        reconst_loss = F.mse_loss(out, data_i, size_average=False)
                    kl_divergence = torch.sum(0.5 * (mu ** 2 + torch.exp(log_var) - log_var - 1))
                
                    C_max = Variable(torch.FloatTensor([args.C_max]).to(device))
                    C = torch.clamp(C_max/args.C_stop_iter*i, 0, C_max.data[0])
                    
                    reconst_loss /= 2048*19
                    kl_divergence /= 2048*19
                    total_loss = reconst_loss + args.gamma*(kl_divergence-C).abs()
                
                elif args.model == "Beta_TCVAE" or args.model == "conv_Beta_TCVAE":
                    dataset_size = 14 * 2048
                    total_loss, reconst_loss, kl_divergence, tc_loss, mi_loss = TCVAE_loss_function(data_i, out, mu, log_var, z, dataset_size, batch_iter=i, 
 anneal_steps=args.anneal_steps, alpha=args.alpha, beta=args.beta, gamma=args.gamma, train=False)


                print(f"Val == Epoch: {epoch}/{args.num_epochs} | Batch: {i} | Recons: {reconst_loss:.6f} | KLD: {kl_divergence:.6f}")

            plot_latent_space(z.detach().cpu().numpy(), args.output_dir, epoch, name="val")

    return model
