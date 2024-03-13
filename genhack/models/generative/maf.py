"""
https://pytorch-lightning.readthedocs.io/en/stable/notebooks/course_UvA-DL/09-normalizing-flows.html
"""
import math

from nflows.flows import MaskedAutoregressiveFlow
from torch import nn
import torch


class MAF(nn.Module):

    def __init__(self,
                 n_layers,
                 n_dim,
                 n_latent_dim,
                 n_hidden_features,
                 n_blocks,
                 ts_model,
                 weights_model,
                 dropout_probability=0.0,
                 use_residual_blocks=True,
                 use_batch_norm=False,
                 use_random_permutations=False,
                 use_random_masks=False,
                 *args, **kwargs):

        """Note that you can disable weighting by using PowerLawWeights with c = 0."""
        super().__init__()
        self.n_layers = n_layers
        self.n_dim = n_dim
        self.n_latent_dim = n_latent_dim
        self.n_hidden_features = n_hidden_features
        self.n_blocks = n_blocks
        self.ts_model = ts_model
        self.weights_model = weights_model
        self.use_batch_norm = use_batch_norm
        self.use_residual_blocks = use_residual_blocks
        self.use_random_permutations = use_random_permutations
        self.use_random_masks = use_random_masks
        self.dropout_probability = dropout_probability

        # initialize flow
        self.flow = MaskedAutoregressiveFlow(features=self.n_dim,
                                             num_layers=self.n_layers,
                                             hidden_features=self.n_hidden_features,
                                             num_blocks_per_layer=self.n_blocks,
                                             use_residual_blocks=self.use_residual_blocks,
                                             use_random_permutations=self.use_random_permutations,
                                             use_random_masks=self.use_random_masks,
                                             dropout_probability=self.dropout_probability,
                                             batch_norm_within_layers=self.use_batch_norm)

    def forward(self, inputs):
        sst, time = inputs
        return sst, time

    # @todo this is hardcoded for now, remove t_min, t_max from sampling
    def sample(self, noise, t_min=0.75, t_max=1.):
        # noise and time samples
        norm_cdf = lambda x: 0.5 * (1 + torch.erf(x / math.sqrt(2)))
        time = t_min + (t_max - t_min) * norm_cdf(noise[:, 6])
        noise = noise[:, :6]

        samples = torch.squeeze(self.flow._transform.inverse(noise)[0])

        # entrend
        # @todo currently we have ts_model + generative_model
        # one should consider conditional models too
        if self.ts_model is not None:
            samples += self.ts_model(time)

        return samples

    def loss(self, *args, **kwargs):
        sst, time = args
        weights = self.weights_model(time)
        return {'loss': torch.mean(-weights * self.flow.log_prob(inputs=sst))}
