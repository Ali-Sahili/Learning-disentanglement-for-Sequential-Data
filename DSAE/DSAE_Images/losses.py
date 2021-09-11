
import math
import torch
import numpy as np
import torch.nn.functional as F





def loss_fn(original_seq,recon_seq,f_mean,f_logvar,z_mean,z_logvar):
    # print(recon_seq.shape, original_seq.shape);assert(0)
    mse = F.mse_loss(recon_seq,original_seq,reduction='sum');
    kld_f = -0.5 * torch.sum(1 + f_logvar - torch.pow(f_mean,2) - torch.exp(f_logvar))
    kld_z = -0.5 * torch.sum(1 + z_logvar - torch.pow(z_mean,2) - torch.exp(z_logvar))

    return mse + kld_f + kld_z, mse, kld_f, kld_z


def log_density_gaussian(x, mu, logvar):
        """
        Computes the log pdf of the Gaussian with parameters mu and logvar at x
        :param x: (Tensor) Point at whichGaussian PDF is to be evaluated
        :param mu: (Tensor) Mean of the Gaussian distribution
        :param logvar: (Tensor) Log variance of the Gaussian distribution
        :return:
        """
        norm = - 0.5 * (math.log(2 * math.pi) + logvar)
        log_density = norm - 0.5 * ((x - mu) ** 2 * torch.exp(-logvar))
        return log_density

def mutual_inf(model, x, f, f_mean, f_logvar, z, z_mean, z_logvar, device):

    dataset_size = 6680

    n_frames = x.shape[1]
    b_size = x.shape[0]

    # Mutual Information Regularization
    z = z.view(b_size * n_frames, -1)
    z_mean = z_mean.view(b_size * n_frames, -1)
    z_logvar = z_logvar.view(b_size * n_frames, -1)

    #log_q_f_x = log_density_gaussian(f, f_mean, f_logvar).sum(dim = 1)
    log_q_z_x = log_density_gaussian(z, z_mean, z_logvar).sum(dim = 1)

    f_expand = f.unsqueeze(1).expand(-1,model.frames,model.f_dim)
    f_mean_expand = f_mean.unsqueeze(1).expand(-1,model.frames,model.f_dim)
    f_logvar_expand = f_logvar.unsqueeze(1).expand(-1,model.frames,model.f_dim)

    f = f_expand.contiguous().view(b_size * n_frames, -1)
    f_mean = f_mean_expand.contiguous().view(b_size * n_frames, -1)
    f_logvar = f_logvar_expand.contiguous().view(b_size * n_frames, -1)

    log_q_f_x = log_density_gaussian(f, f_mean, f_logvar).sum(dim = 1)

    #print(f.shape,z.shape);assert(0)
    zf = torch.cat([f, z], dim=-1)
    zf_mean = torch.cat([f_mean, z_mean], dim=-1)
    zf_logvar = torch.cat([f_logvar, z_logvar], dim=-1)
    #print(zf.shape, zf_mean.shape, zf_logvar.shape)

    mat_log_q_zf = log_density_gaussian( zf.view(b_size*n_frames, 1, -1), 
                                         zf_mean.view(1, b_size*n_frames, -1),
                                         zf_logvar.view(1, b_size*n_frames, -1))

    # Reference
    # [1] https://github.com/YannDubs/disentangling-vae/blob/535bbd2e9aeb5a200663a4f82f1d34e084c4ba8d/disvae/utils/math.py#L54
    bframes = b_size*n_frames
    strat_weight = (dataset_size - bframes + 1) / (dataset_size * (bframes - 1))
    assert strat_weight > 0
    importance_weights = torch.Tensor(bframes, bframes).fill_(1 / (bframes -1)).to(device)
    importance_weights.view(-1)[::bframes] = 1 / dataset_size
    importance_weights.view(-1)[1::bframes] = strat_weight
    importance_weights[bframes - 2, 0] = strat_weight
    log_importance_weights = importance_weights.log()

    mat_log_q_zf += log_importance_weights.view(bframes, bframes, 1)

    log_q_zf = torch.logsumexp(mat_log_q_zf.sum(2), dim=1, keepdim=False)

    #print(log_q_f_x.shape, log_q_z_x.shape, log_q_zf.shape)
    loss_MI = (log_q_f_x + log_q_z_x - log_q_zf).mean()
    #print("Mutual Information loss: ", loss_MI)

    return loss_MI


def cyclic_loss(model, data, z, f, recon_x, device):

    z_rand = torch.rand(z.shape).to(device)
    f_rand = torch.rand(f.shape).to(device)

    z_1 = z.detach() + z_rand
    f_1 = f.detach() + f_rand

    f_expand = f_1.unsqueeze(1).expand(-1, model.frames, model.f_dim)
    zf = torch.cat((z_1,f_expand),dim=2)
    recon_x_1 = model.decode_frames(zf)

    _,_, f_2, _,_, z_2, _ = model(recon_x_1)

    z_3 = z_2.detach() - z_rand
    f_3 = f_2.detach() - f_rand

    f_expand = f_3.unsqueeze(1).expand(-1, model.frames, model.f_dim)
    zf = torch.cat((z_3,f_expand),dim=2)
    new_recon_x = model.decode_frames(zf)

    loss = F.mse_loss(new_recon_x, recon_x, reduction='sum');

    return loss
