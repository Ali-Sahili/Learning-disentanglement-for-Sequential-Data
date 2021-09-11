import torch
import numpy as np
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter


from losses import loss_fn, mutual_inf, cyclic_loss



class Trainer(object):
    def __init__(self,model,device,train,trainloader,epochs,batch_size,learning_rate):
        self.trainloader = trainloader
        self.nc = 1
        self.train = train
        self.start_epoch = 0
        self.epochs = epochs
        self.device = device
        self.batch_size = batch_size
        self.model = model
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
           MI_loss = []
           print("Running Epoch : {}".format(epoch+1))
           for i,data in enumerate(self.trainloader,1):
               #print(data.shape)
               #data = data.transpose_(3, 4).transpose_(2, 3)
               #print(data.shape);assert(0)
               data = data.to(device)
               self.optimizer.zero_grad()
               f_mean,f_logvar,f,z_mean,z_logvar,z,recon_x = self.model(data)
               loss, mse, kld_f, kld_z = loss_fn(data,recon_x,f_mean,f_logvar,z_mean,z_logvar)

               # Static Consistency Constraint
               margin = 1.
               idx = torch.randperm(data.shape[0])
               data_pos = data[idx].view(data.size())
               _,_, f_pos, _,_, _, _ = self.model(data_pos)

               loss_SCC = F.mse_loss(f, f_pos) + margin

               lamda = 100.
               loss += lamda*loss_SCC

               # Cyclic loss
               cyc_loss = cyclic_loss(self.model, data, z, f, recon_x, self.device)
               lamda_2 = 0.1
               loss += lamda_2 * cyc_loss

               # Mutual Information
               mi_loss = mutual_inf(self.model, data, f, f_mean, f_logvar,
                                                z, z_mean, z_logvar,
                                                self.device)
               lamda_3 = 10.
               loss += lamda_3 * mi_loss

               loss.backward()
               self.optimizer.step()

               losses.append(loss.item())
               kld_fs.append(kld_f.item())
               kld_zs.append(kld_z.item())
               MI_loss.append(mi_loss.item())
               mse_loss.append(mse)

           meanloss = np.mean(losses)
           meanf = np.mean(kld_fs)
           meanz = np.mean(kld_zs)
           mean_mse = np.mean(kld_zs)
           mean_mi = np.mean(MI_loss)

           # logging losses
           writer.add_scalar('training loss', meanloss, epoch)
           writer.add_scalar('KL of f', meanf, epoch)
           writer.add_scalar('KL of z', meanz, epoch)
           writer.add_scalar('Reconstruction loss', mean_mse, epoch)
           writer.add_scalar('MI loss', mean_mi, epoch)


           print("Epoch {} : Average Loss: {}".format(epoch+1,meanloss))

           if epoch%2 == 0:
               print("Saving...")
               filepath = "models/" + str(epoch) + ".pth"
               with open(filepath, 'wb+') as f:
                   torch.save(self.model.state_dict(), f)

           self.model.train()
       print("Training is complete")
