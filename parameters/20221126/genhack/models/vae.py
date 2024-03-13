from torch import nn
from torch.nn import functional as F
import torch


class VAE(nn.Module):

    def __init__(self, n_dim, n_latent_dim, n_hidden_dims, kld_weight, *args, **kwargs):
        super().__init__()
        self.n_dim = n_dim
        self.n_latent_dim = n_latent_dim
        self.n_hidden_dims = n_hidden_dims
        self.kld_weight = kld_weight

        # build encoder

        in_channels = n_dim

        modules = []
        for dim in self.n_hidden_dims:
            modules.append(nn.Sequential(
                nn.Linear(in_channels, dim),
                nn.BatchNorm1d(dim),
                nn.LeakyReLU(),
            ))

            in_channels = dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(self.n_hidden_dims[-1], self.n_latent_dim)
        self.fc_log_var = nn.Linear(self.n_hidden_dims[-1], self.n_latent_dim)

        # build decoder

        modules = []
        self.decoder_input = nn.Linear(self.n_latent_dim, self.n_hidden_dims[-1])

        for dim in self.n_hidden_dims[::-1]:
            modules.append(nn.Sequential(
                nn.Linear(in_channels, dim),
                nn.BatchNorm1d(dim),
                nn.LeakyReLU(),
            ))

            in_channels = dim

        self.decoder = nn.Sequential(*modules)

    def encode(self, input):
        result = self.encoder(input)
        mu, log_var = self.fc_mu(result), self.fc_log_var(result)
        return mu, log_var

    def decode(self, z):
        result = self.decoder_input(z)
        result = self.decoder(result)
        return result

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input):
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), input, mu, log_var

    def loss(self, *args, **kwargs):
        reconstruction, input, mu, log_var = args
        reconstruction_loss = F.mse_loss(reconstruction, input)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu**2 - log_var.exp(), dim=1), dim=0)
        loss = reconstruction_loss + self.kld_weight + kld_loss
        return {'loss': loss, 'reconstruction_loss': reconstruction_loss, 'kld_loss': kld_loss}

    def sample(self, noise):
        return self.decode(noise)
