"""
- “Deep Kalman Filter” arXiv, 2015, Rahul G.Krishnan et al.
- "Structured Inference Networks for Nonlinear State Space Models" AAAI, 2017, Rahul G.Krishnan et al.
DKF refers to the deep Markov model in the second paper, which has two possibilities:
- with only backwrad RNN in inference, it's a Deep Kalman Smoother (DKS),
- with bi-directional RNN in inference, it's a ST-LR
To have consistant expression comparing with other models we change some functions' name:
Emissino Function -> Generation
Gated Transition Fucntion -> Prior
"""


from torch import nn
import torch
import torch.nn.functional as F
from collections import OrderedDict


class DKF(nn.Module):

    def __init__(self, x_dim, z_dim=16, activation='tanh',
                 dense_x_gx=[], dim_RNN_gx=128, num_RNN_gx=1, bidir_gx=False,
                 dense_ztm1_g=[], dense_g_z=[],
                 dense_z_x=[128,128],
                 dropout_p = 0, beta=1, device='cpu'):

        super().__init__()
        ### General parameters  
        self.x_dim = x_dim
        self.y_dim = x_dim
        self.z_dim = z_dim
        self.dropout_p = dropout_p
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            raise SystemExit('Wrong activation type!')
        self.device = device
        ### Inference
        self.dense_x_gx = dense_x_gx
        self.dim_RNN_gx = dim_RNN_gx
        self.num_RNN_gx = num_RNN_gx
        self.bidir_gx = bidir_gx
        self.dense_ztm1_g = dense_ztm1_g
        self.dense_g_z = dense_g_z
        ### Generation x
        self.dense_z_x = dense_z_x
        ### Beta-loss
        self.beta = beta

        self.build()


    def build(self):
    
        ###################
        #### Inference ####
        ###################
        # 1. x_t to g_tˆx
        dic_layers = OrderedDict()
        if len(self.dense_x_gx) == 0:
            dim_x_gx = self.x_dim
            dic_layers['Identity'] = nn.Identity()
        else:
            dim_x_gx = self.dense_x_gx[-1]
            for n in range(len(self.dense_x_gx)):
                if n == 0:
                    dic_layers['linear'+str(n)] = nn.Linear(self.x_dim, self.dense_x_gx[n])
                else:
                    dic_layers['linear'+str(n)] = nn.Linear(self.dense_x_gx[n-1], self.dense_x_gx[n])
                dic_layers['activation'+str(n)] = self.activation
                dic_layers['dropout'+str(n)] = nn.Dropout(p=self.dropout_p)
        self.mlp_x_gx = nn.Sequential(dic_layers)
        self.rnn_gx = nn.LSTM(dim_x_gx, self.dim_RNN_gx, self.num_RNN_gx, bidirectional=self.bidir_gx)
        # 2. g_tˆx and z_tm1 to g_t
        dic_layers = OrderedDict()
        n = len(self.dense_x_gx)
        if len(self.dense_ztm1_g) == 0:
            dic_layers['linear_last'] = nn.Linear(self.z_dim, self.dim_RNN_gx)
            dic_layers['activation_last'] = self.activation
            dic_layers['dropout_last'+str(n)] = nn.Dropout(p=self.dropout_p)
        else:
            for n in range(len(self.dense_ztm1_g)):
                if n == 0:
                    dic_layers['linear'+str(n)] = nn.Linear(self.z_dim, self.dense_ztm1_g[n])
                else:
                    dic_layers['linear'+str(n)] = nn.Linear(self.dense_ztm1_g[n-1], self.dense_ztm1_g[n])
                dic_layers['activation'+str(n)] = self.activation
                dic_layers['dropout'+str(n)] = nn.Dropout(p=self.dropout_p)
            dic_layers['linear_last'] = nn.Linear(self.dense_ztm1_g[-1], self.dim_RNN_gx)
            dic_layers['activation_last'] = self.activation
            dic_layers['dropout_last'+str(n)] = nn.Dropout(p=self.dropout_p)
        self.mlp_ztm1_g = nn.Sequential(dic_layers)
        # 3. g_t to z_t
        dic_layers = OrderedDict()
        if len(self.dense_g_z) == 0:
            dim_g_z = self.dim_RNN_gx
            dic_layers['Identity'] = nn.Identity()
        else:
            dim_g_z = self.dense_g_z[-1]
            for n in range(len(self.dense_g_z)):
                if n == 0:
                    dic_layers['linear'+str(n)] = nn.Linear(self.dim_RNN_gx, self.dense_g_z[n])
                else:
                    dic_layers['linear'+str(n)] = nn.Linear(self.dense_g_z[n-1], self.dense_g_z[n])
                dic_layers['activation'+str(n)] = self.activation
                dic_layers['dropout'+str(n)] = nn.Dropout(p=self.dropout_p)
        self.mlp_g_z = nn.Sequential(dic_layers)
        self.inf_mean = nn.Linear(dim_g_z, self.z_dim)
        self.inf_logvar = nn.Linear(dim_g_z, self.z_dim)

        ######################
        #### Generation z ####
        ######################
        # 1. Gating Unit
        dic_layers = OrderedDict()
        dic_layers['linear1'] = nn.Linear(self.z_dim, self.z_dim)
        dic_layers['ReLU'] = nn.ReLU()
        dic_layers['linear2'] = nn.Linear(self.z_dim, self.z_dim)
        dic_layers['Sigmoid'] = nn.Sigmoid()
        self.mlp_gate = nn.Sequential(dic_layers)
        # 2. Proposed mean
        dic_layers = OrderedDict()
        dic_layers['linear1'] = nn.Linear(self.z_dim, self.z_dim)
        dic_layers['ReLU'] = nn.ReLU()
        dic_layers['linear2'] = nn.Linear(self.z_dim, self.z_dim)
        self.mlp_z_prop = nn.Sequential(dic_layers)
        # 3. Prior
        self.prior_mean = nn.Linear(self.z_dim, self.z_dim)
        self.prior_logvar = nn.Sequential(nn.ReLU(),
                                          nn.Linear(self.z_dim, self.z_dim),
                                          nn.Softplus())
        
        ######################
        #### Generation x ####
        ######################
        dic_layers = OrderedDict()
        if len(self.dense_z_x) == 0:
            dim_z_x = self.z_dim
            dic_layers['Identity'] = nn.Identity()
        else:
            dim_z_x = self.dense_z_x[-1]
            for n in range(len(self.dense_z_x)):
                if n == 0:
                    dic_layers['linear'+str(n)] = nn.Linear(self.z_dim, self.dense_z_x[n])
                else:
                    dic_layers['linear'+str(n)] = nn.Linear(self.dense_z_x[n-1], self.dense_z_x[n])
                dic_layers['activation'+str(n)] = self.activation
                dic_layers['dropout'+str(n)] = nn.Dropout(p=self.dropout_p)
        self.mlp_z_x = nn.Sequential(dic_layers)
        self.gen_logvar = nn.Linear(dim_z_x, self.y_dim)


    def reparameterization(self, mean, logvar):

        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        
        return torch.addcmul(mean, eps, std)


    def inference(self, x):
        
        seq_len = x.shape[0]
        batch_size = x.shape[1]

        # Create variable holder and send to GPU if needed
        z_mean = torch.zeros((seq_len, batch_size, self.z_dim)).to(self.device)
        z_logvar = torch.zeros((seq_len, batch_size, self.z_dim)).to(self.device)
        z = torch.zeros((seq_len, batch_size, self.z_dim)).to(self.device)
        z_t = torch.zeros((batch_size, self.z_dim)).to(self.device)

        # 1. x_t to g_t, g_t and z_tm1 to z_t
        x_g = self.mlp_x_gx(x)
        if self.bidir_gx:
            g, _ = self.rnn_gx(x_g)
            g = g.view(seq_len, batch_size, 2, self.dim_RNN_gx)
            g_forward = g[:,:,0,:]
            g_backward = g[:,:,1,:]
            for t in range(seq_len):
                g_t = (self.mlp_ztm1_g(z_t) + g_forward[t,:,:] + g_backward[t,:,:]) / 3
                g_z = self.mlp_g_z(g_t)
                z_mean[t,:,:] = self.inf_mean(g_z)
                z_logvar[t,:,:] = self.inf_logvar(g_z)
                z_t = self.reparameterization(z_mean[t,:,:], z_logvar[t,:,:]) 
                z[t,:,:] = z_t
        else:
            g, _ = self.rnn_gx(torch.flip(x_g, [0]))
            g = torch.flip(g, [0])
            for t in range(seq_len):
                g_t = (self.mlp_ztm1_g(z_t) + g[t,:,:]) / 2
                g_z = self.mlp_g_z(g_t)
                z_mean[t,:,:] = self.inf_mean(g_z)
                z_logvar[t,:,:] = self.inf_logvar(g_z)
                z_t = self.reparameterization(z_mean[t,:,:], z_logvar[t,:,:])
                z[t,:,:] = z_t

        return z, z_mean, z_logvar
    
    
    def generation_z(self, z_tm1):

        gate = self.mlp_gate(z_tm1)
        z_prop = self.mlp_z_prop(z_tm1)
        z_mean_p = (1 - gate) * self.prior_mean(z_tm1) + gate * z_prop
        z_var_p = self.prior_logvar(z_prop)
        z_logvar_p = torch.log(z_var_p) # consistant with other models

        return z_mean_p, z_logvar_p


    def generation_x(self, z):
        
        # 1. z_t to y_t
        log_y = self.mlp_z_x(z)
        log_y = self.gen_logvar(log_y)
        y = torch.exp(log_y)
        
        return y
    

    def forward(self, x):
        
        # train input: (batch_size, x_dim, seq_len)
        # test input:  (x_dim, seq_len)
        # need input:  (seq_len, batch_size, x_dim)
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
        x = x.permute(-1, 0, 1)

        seq_len = x.shape[0]
        batch_size = x.shape[1]

        # main part 
        z, z_mean, z_logvar = self.inference(x)
        z_0 = torch.zeros(1, batch_size, self.z_dim).to(self.device)
        z_tm1 = torch.cat([z_0, z[:-1, :,:]], 0)
        z_mean_p, z_logvar_p = self.generation_z(z_tm1)
        y = self.generation_x(z)

        # calculate loss
        loss_tot, loss_recon, loss_KLD = self.get_loss(x, y, z_mean, z_logvar,
                                                        z_mean_p, z_logvar_p,
                                                        seq_len, batch_size, self.beta)
        self.loss = (loss_tot, loss_recon, loss_KLD)
        
        # output of NN:    (seq_len, batch_size, dim)
        # output of model: (batch_size, dim, seq_len) or (dim, seq_len)
        self.y = y.permute(1,-1,0).squeeze()
        self.z = z.permute(1,-1,0).squeeze()
        self.z_mean = z_mean.permute(1,-1,0).squeeze()
        self.z_logvar = z_logvar.permute(1,-1,0).squeeze()
        self.z_mean_p = z_mean_p.permute(1,-1,0).squeeze()
        self.z_logvar_p = z_logvar_p.permute(1,-1,0).squeeze()

        return self.y, self.z, loss_tot, loss_recon, loss_KLD


    def get_loss(self, x, y, z_mean, z_logvar, z_mean_p, z_logvar_p, seq_len, batch_size, beta=1):

        #loss_recon = torch.sum( x/y - torch.log(x/y) - 1)
        loss_recon = F.mse_loss(x, y, size_average=False)
        loss_KLD = -0.5 * torch.sum(z_logvar - z_logvar_p 
                - torch.div(z_logvar.exp() + (z_mean - z_mean_p).pow(2), z_logvar_p.exp()))
        
        loss_recon = loss_recon / (batch_size * seq_len)
        loss_KLD = loss_KLD / (batch_size * seq_len)
        loss_tot = loss_recon + beta * loss_KLD

        return loss_tot, loss_recon, loss_KLD

