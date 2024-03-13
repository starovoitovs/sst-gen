import math

import torch
from nflows.distributions import StandardNormal
from nflows.flows import Flow
from nflows.transforms import RandomPermutation, ReversePermutation, MaskedAffineAutoregressiveTransform, BatchNorm, CompositeTransform
from torch import nn

from genhack.utils import DEVICE


class CNF(nn.Module):

    def __init__(self,
                 n_layers,
                 n_dim,
                 n_latent_dim,
                 n_hidden_features,
                 n_condition_features,
                 n_blocks,
                 dropout_probability=0.0,
                 use_residual_blocks=True,
                 use_batch_norm_within_layers=False,
                 use_batch_norm_between_layers=False,
                 use_random_permutations=False,
                 use_random_masks=False,
                 *args, **kwargs):

        super().__init__()

        """Note that you can disable weighting by using PowerLawWeights with c = 0."""
        self.n_layers = n_layers
        self.n_dim = n_dim
        self.n_latent_dim = n_latent_dim
        self.n_hidden_features = n_hidden_features
        self.n_condition_features = n_condition_features
        self.n_blocks = n_blocks
        self.use_batch_norm_within_layers = use_batch_norm_within_layers
        self.use_batch_norm_between_layers = use_batch_norm_between_layers
        self.use_residual_blocks = use_residual_blocks
        self.use_random_permutations = use_random_permutations
        self.use_random_masks = use_random_masks
        self.dropout_probability = dropout_probability

        # initialize flow
        if self.use_random_permutations:
            permutation_constructor = RandomPermutation
        else:
            permutation_constructor = ReversePermutation

        layers = []
        for _ in range(self.n_layers):
            layers.append(permutation_constructor(self.n_dim))
            layers.append(
                MaskedAffineAutoregressiveTransform(
                    features=n_dim,
                    hidden_features=n_hidden_features,
                    context_features=n_condition_features,
                    num_blocks=n_blocks,
                    use_residual_blocks=use_residual_blocks,
                    random_mask=use_random_masks,
                    dropout_probability=dropout_probability,
                    use_batch_norm=self.use_batch_norm_within_layers,
                )
            )

            if self.use_batch_norm_between_layers:
                layers.append(BatchNorm(self.n_dim))

        self.flow = Flow(
            transform=CompositeTransform(layers),
            distribution=StandardNormal([self.n_dim]),
        )

    def forward(self, inputs):
        sst, position, time = inputs
        return sst, position, time

    def sample(self, noise, position, time=None):

        if time is None:
            time = torch.linspace(0.7453, 1., len(noise), device=DEVICE)

        noise = noise[:, :self.n_dim]
        position = position.repeat(len(noise)).reshape(len(noise), -1)
        context = torch.cat([time[:, None], position], dim=1)

        return torch.squeeze(self.flow._transform.inverse(noise, context=context)[0])

    def loss(self, *args, **kwargs):
        sst, position, time = args
        context = torch.cat([time[:, None], position], dim=1)
        return {'loss': torch.mean(-self.flow.log_prob(inputs=sst, context=context))}
