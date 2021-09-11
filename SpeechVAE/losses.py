import math
import torch
import torch.nn.functional as F

def log_density_gaussian(x, mu, logvar):
    """
    Computes the log pdf of the Gaussian with parameters mu and logvar at x
    """
    norm = - 0.5 * (math.log(2 * math.pi) + logvar)
    log_density = norm - 0.5 * ((x - mu) ** 2 * torch.exp(-logvar))
    return log_density

# Computes the beta-TCVAE loss function
def TCVAE_loss_function(input, recons, mu, log_var, z, dataset_size, batch_iter, 
                           anneal_steps=200., alpha=1., beta=6., gamma=1., train=True):
    """
    KL(N(\mu,\sigma),N(0,1)) = \log\frac{1}{\sigma} + \frac{\sigma^2 +\mu^2}{2} - \frac{1}{2}
    """
    weight = 1 # Account for the minibatch samples from the dataset
    batch_size, z_dim = z.shape
    
    recons_loss = F.mse_loss(recons, input, reduction='sum')/(19*batch_size)

    log_q_zx = log_density_gaussian(z, mu, log_var).sum(dim = 1)

    zeros = torch.zeros_like(z)
    log_p_z = log_density_gaussian(z, zeros, zeros).sum(dim = 1)

    mat_log_q_z = log_density_gaussian(      z.view(batch_size, 1, z_dim),
                                            mu.view(1, batch_size, z_dim),
                                       log_var.view(1, batch_size, z_dim))

    # Reference
    # [1] https://github.com/YannDubs/disentangling-vae/blob/535bbd2e9aeb5a200663a4f82f1d34e084c4ba8d/disvae/utils/math.py#L54
    strat_weight = (dataset_size - batch_size + 1) / (dataset_size * (batch_size - 1))
    importance_weights = torch.Tensor(batch_size, batch_size).fill_(1 / (batch_size -1)).to(input.device)
    importance_weights.view(-1)[::batch_size] = 1 / dataset_size
    importance_weights.view(-1)[1::batch_size] = strat_weight
    importance_weights[batch_size - 2, 0] = strat_weight
    log_importance_weights = importance_weights.log()

    mat_log_q_z += log_importance_weights.view(batch_size, batch_size, 1)

    log_q_z = torch.logsumexp(mat_log_q_z.sum(2), dim=1, keepdim=False)
    log_prod_q_z = torch.logsumexp(mat_log_q_z, dim=1, keepdim=False).sum(1)

    mi_loss  = (log_q_zx - log_q_z).mean()/19
    tc_loss = (log_q_z - log_prod_q_z).mean()/19
    kld_loss = (log_prod_q_z - log_p_z).mean()/19

    if train:
        batch_iter += 1
        anneal_rate = min(0 + 1 * batch_iter / anneal_steps, 1)
    else:
        anneal_rate = 1.

    loss = recons_loss + alpha*mi_loss + weight*(beta*tc_loss + 
                                                              anneal_rate*gamma*kld_loss)
        
    return loss, recons_loss, kld_loss, tc_loss, mi_loss
