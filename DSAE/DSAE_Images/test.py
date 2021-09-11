import os
import numpy as np

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
from torchvision.utils import make_grid, save_image
import imageio

import matplotlib.pyplot as plt
from disVAE import FullQDisentangledVAE, loss_fn

def save_gif(sample, output_dir, gif_name, duration=0.2):
    imgs = []
    for i in range(sample.shape[0]):
        imgs.append(sample[i])
    
    imageio.mimsave(os.path.join(output_dir, gif_name), imgs, duration = duration)

def plot(x, output_dir, name="inputs", show = False):
    grid = make_grid(x[:,0], nrow=8, padding=3)
    npimg = grid.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')
    #plt.axis('off')
    import imageio
    imageio.imwrite(output_dir + "/" + name +".png", np.transpose(npimg, (1,2,0)))
    #plt.savefig(output_dir + "/" + name +".png")
    if show: plt.show()
            
def test(batch_size, test_loader, device, filename="full_q"):

    test_set_size = len(test_loader.dataset)
    print("number of samples: ", test_set_size)

    output_dir = 'results/' + filename
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    duration = 0.2
    
    sample = next(iter(test_loader))[0]
    print(sample.shape)
    
    model = FullQDisentangledVAE(frames=8,f_dim=64,z_dim=32,hidden_dim=512,conv_dim=1024) 
    
    print("loading model ...")
    filepath = "models/" + filename +".pth"
    with open(filepath, 'rb') as f:
        checkpoint = torch.load(f)
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    with torch.no_grad():
        for i, x in enumerate(test_loader):

            x_save = x.clone()
            x_save = x_save.transpose_(2, 3).transpose_(3, 4).clone()
            
            plot(x, output_dir, name="inputs_"+str(i))
            
            #save_gif(x_save[0], output_dir, "in_"+ str(i) +".gif", duration)
            x = x.to(device)
            
            f_mean,f_logvar,f,z_mean,z_logvar,z,recon_x = model(x)
            #print(x.shape, recon_x.shape);assert(0)
            loss, mse, kld_f, kld_z = loss_fn(x, recon_x, f_mean, f_logvar, z_mean, z_logvar)
                       
            print("Batch {} : Average Loss: {} | MSE : {}  | KL of f : {} | KL of z : {}".format(i, loss, mse, kld_f, kld_z))
                
            output = recon_x.detach().cpu()
            output = output.view(batch_size, -1, 3, 64, 64)
            plot(output, output_dir, name="output_"+str(i))
            
            #output = output.transpose_(3, 2).transpose_(4, 3)[0]
            #save_gif(output, output_dir, "out_"+ str(i) +".gif", duration=0.2)
            assert(0)  
                
def latent_taversals(batch_size, test_loader, device, filename="full_q"):

    test_set_size = len(test_loader.dataset)
    print("number of samples: ", test_set_size)

    output_dir = 'results/' + filename
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    duration = 0.2
    
    sample = next(iter(test_loader))[0]
    print(sample.shape)
    
    model = FullQDisentangledVAE(frames=8,f_dim=64,z_dim=32,hidden_dim=512,conv_dim=1024) 
    
    print("loading model ...")
    filepath = "models/" + filename +".pth"
    with open(filepath, 'rb') as f:
        checkpoint = torch.load(f)
        model.load_state_dict(checkpoint)
    
    limit = 2.
    inter = 1/2.5

    interpolation = torch.arange(-limit, limit+0.1, inter)
    #print(interpolation)
    
    model.to(device)
    model.eval()
    with torch.no_grad():
        for i, x in enumerate(test_loader):

            x = x.to(device)
            
            conv_x = model.encode_frames(x)
            f_mean,f_logvar,f_0 = model.encode_f(conv_x)
            z_mean,z_logvar,z_0 = model.encode_z(conv_x,f_0)
            f_expand_0 = f_0.unsqueeze(1).expand(-1,8,64)

            
            for val in interpolation:
                f = f_0 + val
                f_expand = f.unsqueeze(1).expand(-1,8,64)
                z_mean,z_logvar,z = model.encode_z(conv_x,f)
                zf = torch.cat((z,f_expand),dim=2)
                sample = model.decode_frames(zf)
                plot(sample.cpu(), output_dir, name="output_static_latent_"+str(val))
            
            for val in interpolation:
                z = z_0 + val
                zf = torch.cat((z,f_expand_0),dim=2)
                sample = model.decode_frames(zf)
                plot(sample.cpu(), output_dir, name="output_dynamic_latent_"+str(val))
            
            assert(0)
