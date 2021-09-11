import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

def to_img(x, sep=False):
    """
    convert a n-by-T-by-F segment spectrograms to a 2d image
    Args:
        x(np.ndarray): n-by-T-by-F segment spectrograms
        sep(bool): add separators between segments if True
    """
    n = 1
    T, F = x.shape
    if sep:
        min_val = np.min(x)
        sep = np.ones((n, 1, F)) * min_val
        x = np.concatenate([x, sep], axis=1)
        T += 1
    return x.reshape((n * T, F)).transpose()

def plot(x_val, output_dir, img="result", epoch=1, n_segs=8, clim=(-2., 2.)):
 
    nrows = len(x_val)
    
    for i in range(n_segs):
        fig = plt.figure(figsize=(6, 6))
        x_2d = to_img(x_val[i], sep=False)
        im = plt.imshow(x_2d, interpolation="none", origin="lower")
        im.set_clim(*clim)
        #fig.subplots_adjust(hspace=0.2)
        plt.savefig(output_dir + "/" + img + "_Epoch_" + str(epoch) + "_seg_"+ str(i+1) +".png")
        plt.close(fig)
    #plt.show()


def plot_latent_space(latent, out_dir, epoch, name="Train"):
    N = latent.shape[1]
    labels = np.random.randint(N, size=latent.shape[0])
    latent_embedded = TSNE(n_components=2).fit_transform(latent)

    fig = plt.figure(figsize=(8, 6))
    plt.scatter( latent_embedded[:, 0], latent_embedded[:, 1],
                                                           c=labels, marker='o', edgecolor='none')
    plt.colorbar(ticks=range(N))
    plt.grid(True)
    plt.savefig(out_dir + "/latent_space_"+ name + "_" +str(epoch) +".png")
    plt.close(fig)
    #plt.show()
