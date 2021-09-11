import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


class FullQDisentangledVAE(nn.Module):
    def __init__(self,frames,f_dim,z_dim,conv_dim,hidden_dim):
        super(FullQDisentangledVAE,self).__init__()
        self.f_dim = f_dim
        self.z_dim = z_dim
        self.frames = frames
        self.conv_dim = conv_dim
        self.hidden_dim = hidden_dim
        self.nc = 3


        self.f_lstm = nn.LSTM(self.conv_dim, self.hidden_dim, 1,
                bidirectional=True,batch_first=True)
        self.f_mean = nn.Linear(self.hidden_dim*2, self.f_dim)
        self.f_mean_drop = nn.Dropout(0.5)
        self.f_logvar_drop = nn.Dropout(0.5)
        self.f_logvar = nn.Linear(self.hidden_dim*2, self.f_dim)

        self.z_lstm = nn.LSTM(self.conv_dim+self.f_dim, self.hidden_dim, 1,
                 bidirectional=True,batch_first=True)
        self.z_rnn = nn.RNN(self.hidden_dim*2, self.hidden_dim,batch_first=True) 
        self.z_mean = nn.Linear(self.hidden_dim, self.z_dim)
        self.z_logvar = nn.Linear(self.hidden_dim, self.z_dim)
        self.z_mean_drop = nn.Dropout(0.5)
        self.z_logvar_drop = nn.Dropout(0.5)

        self.conv1 = nn.Conv2d(self.nc,256,kernel_size=4,stride=2,padding=1)
        self.conv2 = nn.Conv2d(256,256,kernel_size=4,stride=2,padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.drop2 = nn.Dropout2d(0.5)
        self.conv3 = nn.Conv2d(256,256,kernel_size=4,stride=2,padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.drop3 = nn.Dropout2d(0.5)
        self.conv4 = nn.Conv2d(256,256,kernel_size=4,stride=2,padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.drop4 = nn.Dropout2d(0.5)
        self.conv_fc = nn.Linear(4*4*256,self.conv_dim) #4*4 is size 256 is channels
        self.drop_fc = nn.Dropout(0.5)
        self.bnf = nn.BatchNorm1d(self.conv_dim)

        self.deconv_fc = nn.Linear(self.f_dim+self.z_dim,4*4*256) #4*4 is size 256 is channels
        self.deconv_bnf = nn.BatchNorm1d(4*4*256)
        self.drop_fc_deconv = nn.Dropout(0.5)
        self.deconv4 = nn.ConvTranspose2d(256,256,kernel_size=4,stride=2,padding=1)
        self.dbn4 = nn.BatchNorm2d(256)
        self.drop4_deconv = nn.Dropout2d(0.5)
        self.deconv3 = nn.ConvTranspose2d(256,256,kernel_size=4,stride=2,padding=1)
        self.dbn3 = nn.BatchNorm2d(256)
        self.drop3_deconv = nn.Dropout2d(0.5)
        self.deconv2 = nn.ConvTranspose2d(256,256,kernel_size=4,stride=2,padding=1)
        self.dbn2 = nn.BatchNorm2d(256)
        self.drop2_deconv = nn.Dropout2d(0.5)
        self.deconv1 = nn.ConvTranspose2d(256,self.nc,kernel_size=4,stride=2,padding=1)

        for m in self.modules():
            if isinstance(m,nn.BatchNorm2d) or isinstance(m,nn.BatchNorm1d):
                nn.init.constant_(m.weight,1)
                nn.init.constant_(m.bias,1)
            elif isinstance(m,nn.Conv2d) or isinstance(m,nn.ConvTranspose2d) or isinstance(m,nn.Linear):
                nn.init.kaiming_normal_(m.weight,nonlinearity='relu') #Change nonlinearity to 'leaky_relu' if you switch
        nn.init.xavier_normal_(self.deconv1.weight,nn.init.calculate_gain('tanh'))

    def encode_frames(self,x):
        x = x.contiguous().view(-1,self.nc,64,64) #Batchwise stack the 8 images for applying convolutions parallely
        x = F.leaky_relu(self.conv1(x),0.1) #Remove batchnorm, the encoder must learn the data distribution
        x = self.drop2(F.leaky_relu(self.bn2(self.conv2(x)),0.1))
        x = self.drop3(F.leaky_relu(self.bn3(self.conv3(x)),0.1))
        x = self.drop4(F.leaky_relu(self.bn4(self.conv4(x)),0.1))
        x = x.view(-1,4*4*256) #4*4 is size 256 is channels
        x = self.drop_fc(F.leaky_relu(self.bnf(self.conv_fc(x)),0.1)) 
        x = x.view(-1,self.frames,self.conv_dim)
        return x

    def decode_frames(self,zf):
        x = zf.view(-1,self.f_dim+self.z_dim) #For batchnorm1D to work, the frames should be stacked batchwise
        x = self.drop_fc_deconv(F.leaky_relu(self.deconv_bnf(self.deconv_fc(x)),0.1))
        x = x.view(-1,256,4,4) #The 8 frames are stacked batchwise
        x = self.drop4_deconv(F.leaky_relu(self.dbn4(self.deconv4(x)),0.1))
        x = self.drop3_deconv(F.leaky_relu(self.dbn3(self.deconv3(x)),0.1))
        x = self.drop2_deconv(F.leaky_relu(self.dbn2(self.deconv2(x)),0.1))
        x = torch.tanh(self.deconv1(x)) #Images are normalized to -1,1 range hence use tanh. Remove batchnorm because it should fit the final distribution 
        return x.view(-1,self.frames,self.nc,64,64) #Convert the stacked batches back into frames. Images are 64*64*nc

    def reparameterize(self,mean,logvar):
        if self.training:
            eps = torch.randn_like(logvar)
            std = torch.exp(0.5*logvar)
            z = mean + eps*std
            return z
        else:
            return mean

    def encode_f(self,x):
        lstm_out,_ = self.f_lstm(x)
        mean = self.f_mean(self.f_mean_drop(lstm_out[:,self.frames-1])) #The forward and the reverse are already concatenated
        logvar = self.f_logvar(self.f_logvar_drop(lstm_out[:,self.frames-1])) # TODO: Check if its the correct forward and reverse
        #print("Mean shape for f : {}".format(mean.shape))
        return mean,logvar,self.reparameterize(mean,logvar)

    def encode_z(self,x,f):
        f_expand = f.unsqueeze(1).expand(-1,self.frames,self.f_dim)
        lstm_out,_ = self.z_lstm(torch.cat((x, f_expand), dim=2))
        rnn_out,_ = self.z_rnn(lstm_out)
        mean = self.z_mean(self.z_mean_drop(rnn_out))
        logvar = self.z_logvar(self.z_logvar_drop(rnn_out))
        return mean,logvar,self.reparameterize(mean,logvar)

    def forward(self,x):
        conv_x = self.encode_frames(x)
        f_mean,f_logvar,f = self.encode_f(conv_x)
        z_mean,z_logvar,z = self.encode_z(conv_x,f)
        f_expand = f.unsqueeze(1).expand(-1,self.frames,self.f_dim)
        zf = torch.cat((z,f_expand),dim=2)
        recon_x = self.decode_frames(zf)
        return f_mean,f_logvar,f,z_mean,z_logvar,z,recon_x


class Factorized_DisentangledVAE(nn.Module):
    def __init__(self,frames,f_dim,z_dim,conv_dim,hidden_dim):
        super(Factorized_DisentangledVAE,self).__init__()
        self.f_dim = f_dim
        self.z_dim = z_dim
        self.frames = frames
        self.conv_dim = conv_dim
        self.hidden_dim = hidden_dim
        self.nc = 3

        
        self.f_lstm = nn.LSTM(self.conv_dim, self.hidden_dim, 1,
                bidirectional=True,batch_first=True)
        self.f_mean = nn.Linear(self.hidden_dim*2, self.f_dim)
        self.f_mean_drop = nn.Dropout(0.3)
        self.f_logvar_drop = nn.Dropout(0.3)
        self.f_logvar = nn.Linear(self.hidden_dim*2, self.f_dim)

        self.z_lstm = nn.LSTM(self.conv_dim, self.hidden_dim, 1,
                 bidirectional=True,batch_first=True)
        self.z_rnn = nn.RNN(self.hidden_dim*2, self.hidden_dim,batch_first=True) 
        self.z_mean = nn.Linear(self.hidden_dim, self.z_dim)
        self.z_logvar = nn.Linear(self.hidden_dim, self.z_dim)
        self.z_mean_drop = nn.Dropout(0.3)
        self.z_logvar_drop = nn.Dropout(0.3)
        
        self.conv1 = nn.Conv2d(self.nc,256,kernel_size=4,stride=2,padding=1)
        self.conv2 = nn.Conv2d(256,256,kernel_size=4,stride=2,padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.drop2 = nn.Dropout2d(0.4)
        self.conv3 = nn.Conv2d(256,256,kernel_size=4,stride=2,padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.drop3 = nn.Dropout2d(0.4)
        self.conv4 = nn.Conv2d(256,256,kernel_size=4,stride=2,padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.drop4 = nn.Dropout2d(0.4)
        self.conv_fc = nn.Linear(4*4*256,self.conv_dim) #4*4 is size 256 is channels
        self.drop_fc = nn.Dropout(0.4)
        self.bnf = nn.BatchNorm1d(self.conv_dim) 

        self.deconv_fc = nn.Linear(self.f_dim+self.z_dim,4*4*256) #4*4 is size 256 is channels
        self.deconv_bnf = nn.BatchNorm1d(4*4*256)
        self.drop_fc_deconv = nn.Dropout(0.4)
        self.deconv4 = nn.ConvTranspose2d(256,256,kernel_size=4,stride=2,padding=1)
        self.dbn4 = nn.BatchNorm2d(256)
        self.drop4_deconv = nn.Dropout2d(0.4) 
        self.deconv3 = nn.ConvTranspose2d(256,256,kernel_size=4,stride=2,padding=1)
        self.dbn3 = nn.BatchNorm2d(256)
        self.drop3_deconv = nn.Dropout2d(0.4)
        self.deconv2 = nn.ConvTranspose2d(256,256,kernel_size=4,stride=2,padding=1)
        self.dbn2 = nn.BatchNorm2d(256)
        self.drop2_deconv = nn.Dropout2d(0.4)
        self.deconv1 = nn.ConvTranspose2d(256,self.nc,kernel_size=4,stride=2,padding=1)

        for m in self.modules():
            if isinstance(m,nn.BatchNorm2d) or isinstance(m,nn.BatchNorm1d):
                nn.init.constant_(m.weight,1)
                nn.init.constant_(m.bias,1)
            elif isinstance(m,nn.Conv2d) or isinstance(m,nn.ConvTranspose2d) or isinstance(m,nn.Linear):
                nn.init.kaiming_normal_(m.weight,nonlinearity='relu') #Change nonlinearity to 'leaky_relu' if you switch
        nn.init.xavier_normal_(self.deconv1.weight,nn.init.calculate_gain('tanh'))
    
    def encode_frames(self,x):
        x = x.contiguous().view(-1,self.nc,64,64) #Batchwise stack the 8 images for applying convolutions parallely
        x = F.leaky_relu(self.conv1(x),0.1) #Remove batchnorm, the encoder must learn the data distribution
        x = self.drop2(F.leaky_relu(self.bn2(self.conv2(x)),0.1))
        x = self.drop3(F.leaky_relu(self.bn3(self.conv3(x)),0.1))
        x = self.drop4(F.leaky_relu(self.bn4(self.conv4(x)),0.1))
        x = x.view(-1,4*4*256) #4*4 is size 256 is channels
        x = self.drop_fc(F.leaky_relu(self.bnf(self.conv_fc(x)),0.1)) 
        x = x.view(-1,self.frames,self.conv_dim)
        return x

    def decode_frames(self,zf):
        x = zf.view(-1,self.f_dim+self.z_dim) #For batchnorm1D to work, the frames should be stacked batchwise
        x = self.drop_fc_deconv(F.leaky_relu(self.deconv_bnf(self.deconv_fc(x)),0.1))
        x = x.view(-1,256,4,4) #The 8 frames are stacked batchwise
        x = self.drop4_deconv(F.leaky_relu(self.dbn4(self.deconv4(x)),0.1))
        x = self.drop3_deconv(F.leaky_relu(self.dbn3(self.deconv3(x)),0.1))
        x = self.drop2_deconv(F.leaky_relu(self.dbn2(self.deconv2(x)),0.1))
        x = torch.tanh(self.deconv1(x)) #Images are normalized to -1,1 range hence use tanh. Remove batchnorm because it should fit the final distribution 
        return x.view(-1,self.frames,self.nc,64,64) #Convert the stacked batches back into frames. Images are 64*64*nc

    def reparameterize(self,mean,logvar):
        if self.training:
            eps = torch.randn_like(logvar)
            std = torch.exp(0.5*logvar)
            z = mean + eps*std
            return z
        else:
            return mean

    def encode_f(self,x):
        lstm_out,_ = self.f_lstm(x)
        mean = self.f_mean(self.f_mean_drop(lstm_out[:,self.frames-1])) #The forward and the reverse are already concatenated
        logvar = self.f_logvar(self.f_logvar_drop(lstm_out[:,self.frames-1])) # TODO: Check if its the correct forward and reverse
        #print("Mean shape for f : {}".format(mean.shape))
        return mean,logvar,self.reparameterize(mean,logvar)
    
    def encode_z(self,x):
        lstm_out,_ = self.z_lstm(x)
        rnn_out,_ = self.z_rnn(lstm_out)
        mean = self.z_mean(self.z_mean_drop(rnn_out))
        logvar = self.z_logvar(self.z_logvar_drop(rnn_out))
        return mean,logvar,self.reparameterize(mean,logvar)

    def forward(self,x):
        conv_x = self.encode_frames(x)
        f_mean,f_logvar,f = self.encode_f(conv_x)
        z_mean,z_logvar,z = self.encode_z(conv_x,f)
        f_expand = f.unsqueeze(1).expand(-1,self.frames,self.f_dim)
        zf = torch.cat((z,f_expand),dim=2)
        recon_x = self.decode_frames(zf)
        return f_mean,f_logvar,f,z_mean,z_logvar,z,recon_x

