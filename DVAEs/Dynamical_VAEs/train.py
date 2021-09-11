import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
from hessian_penalty import hessian_penalty
from plot import plot, plot_latent_space
from Dynamical_VAEs.losses import TCVAE_loss_function

def train(args, tr_iterator, dt_iterator, device):

    if args.model == "Beta_LSTM_VAE" or args.model == "Annealed_LSTM_VAE":
        from Dynamical_VAEs.lstmvae import VAE
        Z_DIM = 32
        model = VAE(z_dim=Z_DIM, log_var_=None, dropout=0, relu=0.2, n_filters=80)
    elif args.model == "SRNN_VAE" or args.model == "SRNN_Beta_TCVAE":
        from Dynamical_VAEs.srnn import SRNN
        Z_DIM = 32
        model = SRNN(x_dim=80, z_dim=Z_DIM, device=device)
    elif args.model == "STORN_VAE":
        from Dynamical_VAEs.storn import STORN
        Z_DIM = 32
        model = STORN(x_dim=80, z_dim=Z_DIM, device=device)
    elif args.model == "VRNN_VAE":
        from Dynamical_VAEs.vrnn import VRNN
        Z_DIM = 32
        model = VRNN(x_dim=80, z_dim=Z_DIM, device=device)
    elif args.model == "Recurrent_VAE":
        from Dynamical_VAEs.recurrent_vae import RVAE
        Z_DIM = 32
        model = RVAE(x_dim = 80, z_dim = Z_DIM, device=device)
    elif args.model == "KVAE":
        from Dynamical_VAEs.kvae import KVAE
        Z_DIM = 16
        model = KVAE(x_dim = 80, a_dim = Z_DIM, scale_reconstruction=args.beta, device=device)
    elif args.model == "DKF":
        from Dynamical_VAEs.dkf import DKF
        Z_DIM = 32
        model = DKF(x_dim=80, z_dim=Z_DIM, beta=args.beta, device=device)
    elif args.model == "DSAE":
        from Dynamical_VAEs.dsae import DSAE
        Z_DIM = 16
        model = DSAE(x_dim = 80, z_dim = Z_DIM, v_dim = Z_DIM, beta=args.beta, device=device)
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
            
            if args.model == "SRNN_VAE" or args.model == "VRNN_VAE" or args.model == "SRNN_Beta_TCVAE":
                x = torch.transpose(data_i, 1, 2)
                out, z_mean, z_logvar, z_mean_p, z_logvar_p, z = model(x)
            elif args.model == "STORN_VAE" or args.model == "Recurrent_VAE":
                x = torch.transpose(data_i, 1, 2)
                out, z, z_mean, z_logvar = model(x)
            elif args.model == "KVAE" or args.model == "DKF":
                x = torch.transpose(data_i, 1, 2)
                out, z, total_loss, reconst_loss, kl_divergence = model(x)
            elif args.model == "DSAE":
                x = torch.transpose(data_i, 1, 2)
                out, z, v, total_loss, reconst_loss, kl_divergence = model(x)
            else:
                out, mu, log_var, z = model(data_i)

            if args.model == "Beta_LSTM_VAE":
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
                
            elif args.model == "Annealed_LSTM_VAE":
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

            elif args.model == "SRNN_VAE" or args.model == "VRNN_VAE":
                total_loss, reconst_loss, kl_divergence = model.get_loss(x, out, z_mean, z_logvar, z_mean_p, z_logvar_p, seq_len=19, batch_size=256, beta=args.beta)
            
            elif args.model == "STORN_VAE" or args.model == "Recurrent_VAE":
                    total_loss, reconst_loss, kl_divergence = model.get_loss(x, out, z_mean, z_logvar, seq_len=19, batch_size=256, beta=args.beta)
            
            elif args.model == "SRNN_Beta_TCVAE":
                dataset_size = 1580 * 256
                out = torch.transpose(out,1,2)
                z_mean = torch.transpose(z_mean,1,2)
                z_logvar = torch.transpose(z_logvar,1,2)
                z_mean_p = torch.transpose(z_mean_p,1,2)
                z_logvar_p = torch.transpose(z_logvar_p,1,2)
                z = torch.transpose(z,1,2)
                
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
        
        """
        # plotting
        if args.model == "SRNN_VAE" or args.model == "STORN_VAE" or args.model == "VRNN_VAE" or args.model == "Recurrent_VAE" or args.model == "KVAE" or args.model == "DKF" or args.model == "DSAE":
            plot(data_i.cpu().numpy(), args.output_dir, img="image", epoch=epoch)
            plot(torch.transpose(out, 1, 2).detach().cpu().numpy(), args.output_dir, img="result", epoch=epoch) 
            plot_latent_space(z.contiguous().view(-1, Z_DIM).detach().cpu().numpy(), args.output_dir, epoch)
            if args.model == "DSAE":
                plot_latent_space(v.contiguous().view(-1, Z_DIM).detach().cpu().numpy(), args.output_dir, epoch+100)
        else:
            plot(data_i.cpu().numpy(), args.output_dir, img="image", epoch=epoch)
            plot(out.detach().cpu().numpy(), args.output_dir, img="result", epoch=epoch) 
            plot_latent_space(z.detach().cpu().numpy(), args.output_dir, epoch)
        """
        if args.save_step:
            print("Saving...")
            filepath = "outputs/models/" + str(epoch) + ".pth"
            with open(filepath, 'wb+') as f:
                torch.save(model.state_dict(), f)

        """
        model.eval()
        with torch.no_grad():
            for i, (x_val, _, _) in enumerate(dt_iterator()):

                data_i = torch.from_numpy(np.stack(x_val, axis=0)).float().to(device)[:,:19]
                
                if args.model == "SRNN_VAE" or args.model == "VRNN_VAE":
                    x = torch.transpose(data_i, 1, 2)
                    out, z_mean, z_logvar, z_mean_p, z_logvar_p, z = model(x)
                elif args.model == "STORN_VAE" or args.model == "Recurrent_VAE":
                    x = torch.transpose(data_i, 1, 2)
                    out, z, z_mean, z_logvar = model(x)
                elif args.model == "KVAE" or args.model == "DKF":
                    x = torch.transpose(data_i, 1, 2)
                    out, z, total_loss, reconst_loss, kl_divergence = model(x)
                elif args.model == "DSAE":
                    x = torch.transpose(data_i, 1, 2)
                    out, z, v, total_loss, reconst_loss, kl_divergence = model(x)
                else:
                    out, mu, log_var, z = model(data_i)

                if args.model == "Beta_LSTM_VAE":
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
                
                elif args.model == "Annealed_LSTM_VAE":
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

                elif args.model == "SRNN_VAE" or args.model == "VRNN_VAE":
                    total_loss, reconst_loss, kl_divergence = model.get_loss(x, out, z_mean, z_logvar, z_mean_p, z_logvar_p, seq_len=19, batch_size=2048, beta=args.beta)
                
                elif args.model == "STORN_VAE" or args.model == "Recurrent_VAE":
                    total_loss, reconst_loss, kl_divergence = model.get_loss(x, out, z_mean, z_logvar, seq_len=19, batch_size=2048, beta=args.beta)
                    
                print(f"Val == Epoch: {epoch}/{args.num_epochs} | Batch: {i} | Recons: {reconst_loss:.6f} | KLD: {kl_divergence:.6f}")

            if args.model == "SRNN_VAE" or args.model == "STORN_VAE" or args.model == "VRNN_VAE" or args.model == "Recurrent_VAE" or args.model == "KVAE" or args.model == "DKF" or args.model == "DSAE":
                plot_latent_space(z.contiguous().view(-1, Z_DIM).detach().cpu().numpy(), args.output_dir, epoch, name="val")
                if args.model == "DSAE":
                    plot_latent_space(v.contiguous().view(-1, Z_DIM).detach().cpu().numpy(), args.output_dir, epoch+100, name="val")
            else:
                plot_latent_space(z.detach().cpu().numpy(), args.output_dir, epoch, name="val")
        """
    return model
