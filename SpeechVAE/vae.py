import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class VAE(nn.Module):
    def __init__(self,  input_size=20*80, h_dim=[4096, 2048, 1024, 512], z_dim=128, 
                        dropout=0.2, relu=0.1):
        super(VAE, self).__init__()

        if relu == 0:
            activation = nn.Tanh()
        else:
            activation = nn.LeakyReLU(relu)

        self.input_size = input_size  # nb_frames * nb_features --> 20x80=1600
        self.h_dim = [input_size] + h_dim
        layers = []
        for i in range(len(self.h_dim)-1):
            layers.append( nn.Sequential(nn.Linear(self.h_dim[i], self.h_dim[i+1]),
                                              nn.Dropout(dropout),
                                              nn.BatchNorm1d(self.h_dim[i+1]),
                                              activation
                                              )
                              )
        self.enc_layers = nn.ModuleList(layers)
 
        self.mu = nn.Linear(self.h_dim[-1], z_dim, bias=False)
        self.log_var = nn.Linear(self.h_dim[-1], z_dim, bias=False)

        layers = [nn.Linear(z_dim, self.h_dim[-1], bias=False)]
        for i in range(len(self.h_dim)-1, 0, -1):
            layers.append( nn.Sequential(nn.Linear(self.h_dim[i], self.h_dim[i-1]),
                                                 nn.Dropout(dropout),
                                                 nn.BatchNorm1d(self.h_dim[i-1]),
                                                 activation
                                                 )
                                  )
        self.dec_layers = nn.ModuleList(layers)   

    def reparameterize(self, mu, log_var):
        if self.training:
            std = log_var.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            if torch.cuda.is_available():
                eps = eps.cuda()
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, input):
        x = input.contiguous().view(-1, self.input_size)
        for i in range(len(self.enc_layers)):
            x = self.enc_layers[i](x)
        mu = self.mu(x)
        log_var = self.log_var(x)

        z = self.reparameterize(mu.float(), log_var.float())
        x = z
        for i in range(len(self.dec_layers)):
            x = self.dec_layers[i](x)
        x = x.contiguous().view(-1, 19, 80)
        return x, mu, log_var, z

