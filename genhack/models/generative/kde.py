import torch
from scipy.stats import gaussian_kde
from torch import nn


class KDE(nn.Module):
    def __init__(self, n_dim, n_latent_dim, bw_method, datamodule, *args, **kwargs):
        super().__init__()
        self.n_dim = n_dim
        self.n_latent_dim = n_latent_dim
        self.bw_method = bw_method

        inputs = datamodule.train_dataset[:][0].T
        self.kde = gaussian_kde(inputs, bw_method=self.bw_method)

    def sample(self, noise):
        return torch.Tensor(self.kde.resample(len(noise)).T)
