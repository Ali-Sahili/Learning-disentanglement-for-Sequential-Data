
from torch.utils.data import Dataset, DataLoader

from train import Trainer
from models import FullQDisentangledVAE, Factorized_DisentangledVAE


class Sprites(Dataset):
    def __init__(self,path,size):
        self.path = path
        self.length = size;

    def __len__(self):
        return self.length

    def __getitem__(self,idx):
        return torch.load(self.path+'/%d.sprite' % (idx+1))




if __name__ == '__main__':
    vae = FullQDisentangledVAE(frames=8,f_dim=64,z_dim=32,hidden_dim=512,conv_dim=1024) 

    batch_size = 8
    train = True # True
    device = torch.device('cuda') # 'cuda:0'

    if train:
        sprites_train = Sprites('dataset/lpc_dataset/train/', 6680)
        train_loader = DataLoader(sprites_train, batch_size=batch_size, 
                                                                  shuffle=True, num_workers=4)
        print("data loaded.")

        trainer = Trainer(vae, device, sprites_train, train_loader, epochs=1000, 
                                                  batch_size=batch_size, learning_rate=0.0002) 
        trainer.load_checkpoint(epoch=None)
        trainer.train_model()

    else:
        sprites_test = Sprites('dataset/lpc_dataset/test/', 870)
        test_loader = DataLoader(sprites_test, batch_size=batch_size, 
                                                                  shuffle=False, num_workers=4) 
        print("data loaded.")

        from test import test, latent_taversals
        test(batch_size, test_loader, device, filename="results/")
        latent_taversals(batch_size, test_loader, device, filename="results/")

