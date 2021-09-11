import os
import torch
import torchvision
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init
import torch.optim as optim
import numpy as np

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt

class FullQDisentangledVAE(nn.Module):
    def __init__(self,input_size,frames,f_dim,z_dim,conv_dim,hidden_dim):
        super(FullQDisentangledVAE,self).__init__()
        self.f_dim = f_dim
        self.z_dim = z_dim
        self.frames = frames
        self.conv_dim = conv_dim
        self.hidden_dim = hidden_dim
        self.input_size = input_size

        
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
        
        self.conv1 = nn.Linear(self.input_size, 256)
        self.conv2 = nn.Linear(256,256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.5)
        self.conv3 = nn.Linear(256,256)
        self.bn3 = nn.BatchNorm1d(256)
        self.drop3 = nn.Dropout(0.5)
        self.conv4 = nn.Linear(256,256)
        self.bn4 = nn.BatchNorm1d(256)
        self.drop4 = nn.Dropout(0.5)
        self.conv_fc = nn.Linear(256,self.conv_dim) #4*4 is size 256 is channels
        self.drop_fc = nn.Dropout(0.5)
        self.bnf = nn.BatchNorm1d(self.conv_dim) 

        self.deconv_fc = nn.Linear(self.f_dim+self.z_dim,256) #4*4 is size 256 is channels
        self.deconv_bnf = nn.BatchNorm1d(256)
        self.drop_fc_deconv = nn.Dropout(0.5)
        self.deconv4 = nn.Linear(256,256)
        self.dbn4 = nn.BatchNorm1d(256)
        self.drop4_deconv = nn.Dropout(0.5) 
        self.deconv3 = nn.Linear(256,256)
        self.dbn3 = nn.BatchNorm1d(256)
        self.drop3_deconv = nn.Dropout(0.5)
        self.deconv2 = nn.Linear(256,256)
        self.dbn2 = nn.BatchNorm1d(256)
        self.drop2_deconv = nn.Dropout(0.5)
        self.deconv1 = nn.Linear(256,self.input_size)

        for m in self.modules():
            if isinstance(m,nn.BatchNorm1d):
                nn.init.constant_(m.weight,1)
                nn.init.constant_(m.bias,1)
            elif isinstance(m,nn.Linear):
                nn.init.kaiming_normal_(m.weight,nonlinearity='relu') #Change nonlinearity to 'leaky_relu' if you switch
        nn.init.xavier_normal_(self.deconv1.weight,nn.init.calculate_gain('tanh'))
    
    def encode_frames(self,x):
        x = x.contiguous().view(-1,self.input_size) #Batchwise stack the 8 images for applying convolutions parallely
        x = F.leaky_relu(self.conv1(x),0.1) #Remove batchnorm, the encoder must learn the data distribution
        x = self.drop2(F.leaky_relu(self.bn2(self.conv2(x)),0.1))
        x = self.drop3(F.leaky_relu(self.bn3(self.conv3(x)),0.1))
        x = self.drop4(F.leaky_relu(self.bn4(self.conv4(x)),0.1))
        x = x.view(-1,256) #4*4 is size 256 is channels
        x = self.drop_fc(F.leaky_relu(self.bnf(self.conv_fc(x)),0.1)) 
        x = x.view(-1,self.frames,self.conv_dim)
        return x

    def decode_frames(self,zf):
        x = zf.view(-1,self.f_dim+self.z_dim) #For batchnorm1D to work, the frames should be stacked batchwise
        x = self.drop_fc_deconv(F.leaky_relu(self.deconv_bnf(self.deconv_fc(x)),0.1))
        x = x.view(-1,256) #The 8 frames are stacked batchwise
        x = self.drop4_deconv(F.leaky_relu(self.dbn4(self.deconv4(x)),0.1))
        x = self.drop3_deconv(F.leaky_relu(self.dbn3(self.deconv3(x)),0.1))
        x = self.drop2_deconv(F.leaky_relu(self.dbn2(self.deconv2(x)),0.1))
        x = torch.tanh(self.deconv1(x)) #Images are normalized to -1,1 range hence use tanh. Remove batchnorm because it should fit the final distribution 
        return x.view(-1,self.frames,self.input_size) #Convert the stacked batches back into frames. Images are 64*64*nc

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

def loss_fn(original_seq,recon_seq,f_mean,f_logvar,z_mean,z_logvar):
    # print(recon_seq.shape, original_seq.shape);assert(0)
    mse = F.mse_loss(recon_seq,original_seq,reduction='sum');
    kld_f = -0.5 * torch.sum(1 + f_logvar - torch.pow(f_mean,2) - torch.exp(f_logvar))
    kld_z = -0.5 * torch.sum(1 + z_logvar - torch.pow(z_mean,2) - torch.exp(z_logvar))
    return mse + kld_f + kld_z, mse, kld_f, kld_z


class Trainer(object):
    def __init__(self,model,device,data_loader,epochs,batch_size,learning_rate):
        self.trainloader = data_loader
        self.start_epoch = 0
        self.epochs = epochs
        self.device = device
        self.batch_size = batch_size
        self.model = model.double()
        self.model.to(device)
        self.learning_rate = learning_rate
        self.optimizer = optim.Adam(self.model.parameters(),self.learning_rate)


    def load_checkpoint(self, epoch):
        try:
            print("Loading Checkpoint ...")
            self.start_epoch = epoch
            filepath = "models/" + str(epoch) +".pth"
            with open(filepath, 'rb') as f:
                checkpoint = torch.load(f)
                model.load_state_dict(checkpoint)
            print("Resuming Training From Epoch {}".format(epoch))
        except:
            print("No Checkpoint Exists At '{}'.Start Fresh Training".format(epoch))
            self.start_epoch = 0

    def train_model(self):
       # TensorBoard
       writer = SummaryWriter('logs/DSAE_Experiment')

       self.model.train()
       for epoch in range(self.start_epoch,self.epochs):
           losses = []
           kld_fs = []
           kld_zs = []
           mse_loss = []
           print("Running Epoch : {}".format(epoch+1))
           for i, data in enumerate(self.trainloader,1):

               data = data.to(self.device)
               self.optimizer.zero_grad()
               f_mean,f_logvar,f,z_mean,z_logvar,z,recon_x = self.model(data)
               loss, mse, kld_f, kld_z = loss_fn(data,recon_x,f_mean,f_logvar,z_mean,z_logvar)
               loss.backward()
               self.optimizer.step()


               losses.append(loss.item())
               kld_fs.append(kld_f.item())
               kld_zs.append(kld_z.item())
               mse_loss.append(mse)

           meanloss = np.mean(losses)
           meanf = np.mean(kld_fs)
           meanz = np.mean(kld_zs)
           mean_mse = np.mean(kld_zs)

           # logging losses
           writer.add_scalar('training loss', meanloss, epoch)
           writer.add_scalar('KL of f', meanf, epoch)
           writer.add_scalar('KL of z', meanz, epoch)
           writer.add_scalar('Reconstruction loss', mean_mse, epoch)


           print("Epoch {} : Average Loss: {}".format(epoch+1,meanloss))

           if epoch%2 == 0:
               print("Saving...")
               filepath = "models/" + str(epoch) + ".pth"
               with open(filepath, 'wb+') as f:
                   torch.save(self.model.state_dict(), f)

           self.model.train()
       print("Training is complete")



class Timit_Dataset(Dataset):

    def __init__(self, filename="features.npy"):
        self.features = np.load(filename)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx]


def load_data(filename="features.npy",batch_size=16):
    data_set = Timit_Dataset(filename=filename)
    print("data set dimension: ", len(data_set))

    data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=True, 
                                          num_workers=4)

    return data_set, data_loader


def reconstruct_audio(data, name, scale, length, sample_rate, fft_size, hopsamp, verbose=True):
    from utils import istft_function, save_audio_to_file
    
    data = data.view(-1, 257)
    data = data.detach().cpu().numpy()
    if verbose: print(data.shape)
        
    x = np.delete(data,range(data.shape[0]-length,data.shape[0]),axis=0)
    if verbose: print(x.shape)
    
    # Save the spectrogram image also.
    plt.clf()
    plt.figure(1)
    plt.imshow(data.T**0.125, origin='lower', cmap=plt.cm.hot, aspect='auto',
                                                       interpolation='nearest')
    plt.colorbar()
    plt.title('Spectrogram used to reconstruct audio')
    plt.xlabel('time index')
    plt.ylabel('frequency bin index')
    plt.savefig('spectrogram_'+name+'.png', dpi=150)
    
    plt.clf()
    plt.figure(2)
    plt.imshow(x.T**0.125, origin='lower', cmap=plt.cm.hot, aspect='auto',
                                                       interpolation='nearest')
    plt.colorbar()
    plt.title('Spectrogram used to reconstruct audio')
    plt.xlabel('time index')
    plt.ylabel('frequency bin index')
    plt.savefig('spectrogram_without_padding_'+name+'.png', dpi=150)
    
    recon_input = istft_function(x, scale, fft_size, hopsamp)
    if verbose: print(recon_input.shape)
      
    save_audio_to_file(recon_input, sample_rate, outfile=name+".wav")



def test(model, device, seq_length, sample_rate, fft_size, hopsamp):
    
    # Loading data
    data_set = Timit_Dataset(filename="features.npy")
    print("data set dimension: ", len(data_set))
    
    scale_data = np.load("scale.npy")
    len_data = np.load("length.npy")
    
    max_length = np.max(len_data)
    batch_size=int(max_length/seq_length)
    data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=False, 
                                                                      num_workers=4)
    #model = model.double()
    
    # Loading checkpoint
    print("Loading Checkpoint ...")
    filepath = "models/108.pth"
    with open(filepath, 'rb') as f:
        checkpoint = torch.load(f)
        model.load_state_dict(checkpoint)

    model.to(device)

    print("Starting testing phase...")
    for i, data in enumerate(data_loader):

        data = data.float().to(device)
        f_mean, f_logvar, f, z_mean, z_logvar, z, recon_x = model(data)
        loss, mse, kld_f, kld_z = loss_fn(data,recon_x,f_mean,f_logvar,z_mean,z_logvar)
 
        print(data.shape);
        
        scale = scale_data[0]
        length = len_data[0]
        
        # Reconstruct audio from input
        reconstruct_audio(data,"in", scale,length,sample_rate,fft_size,hopsamp,verbose=True)
        
        # Reconstruct audio from output
        reconstruct_audio(recon_x,"out",scale,length,sample_rate,fft_size,hopsamp,verbose=True)
        
        
        
        assert(0)

        
if __name__ == '__main__':
    vae = FullQDisentangledVAE(input_size = 257, frames=20, f_dim=64, z_dim=32,
                               hidden_dim=512, conv_dim=1024)

    batch_size = 16
    train = False # True
    device = torch.device('cuda') # 'cuda:0'

    if train:
        data_set, data_loader = load_data(filename="features.npy",
                                         batch_size=batch_size)
        print("data loaded.")

        trainer = Trainer(vae, device, data_loader, epochs=1000,
                               batch_size=batch_size, learning_rate=0.0002)
        trainer.load_checkpoint(epoch=None)
        trainer.train_model()
    else:
        sample_rate = 16000
        fft_size = 512
        hopsamp = fft_size // 8
        seq_length = 20
        test(vae, device, seq_length, sample_rate, fft_size, hopsamp)
