import torch
from denoising_diffusion_pytorch import Unet1D, GaussianDiffusion1D
from torch import nn, optim
from tqdm import tqdm


class LDM(nn.Module):

    def __init__(self, n_dim, n_latent_dim, n_timesteps, n_unet_dim, n_resnet_block_groups, *args, **kwargs):
        super().__init__()
        self.n_dim = n_dim
        self.n_latent_dim = n_latent_dim
        self.n_timesteps = n_timesteps
        self.n_unet_dim = n_unet_dim
        self.n_resnet_block_groups = n_resnet_block_groups

        model = Unet1D(
            dim=self.n_unet_dim,
            dim_mults=(1, 2),
            channels=1,
            resnet_block_groups=self.n_resnet_block_groups,
        )

        self.diffusion = GaussianDiffusion1D(
            model,
            seq_length=self.n_dim,
            timesteps=self.n_timesteps,
            objective='pred_v',
        )

    def forward(self, inputs):
        return [inputs]

    def sample(self, noise):

        samples = noise[:, None, :]
        x_start = None

        for t in tqdm(reversed(range(0, self.diffusion.num_timesteps)), desc='sampling loop time step', total=self.diffusion.num_timesteps):
            self_cond = x_start if self.diffusion.self_condition else None
            samples, x_start = self.diffusion.p_sample(samples, t, self_cond, clip_denoised=False)

        return torch.squeeze(samples)

    def loss(self, *args, **kwargs):
        input = args[0]
        return {'loss': self.diffusion(input[:, None, :])}
