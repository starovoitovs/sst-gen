from nflows.transforms import MaskedAffineAutoregressiveTransform, CompositeTransform, AffineCouplingTransform
from nflows.transforms.permutations import ReversePermutation, RandomPermutation
from nflows.distributions import StandardNormal
from nflows.flows import Flow
from torch import nn
from torch.nn import functional as F


class MyMaskedAffineAutoregressiveTransform(MaskedAffineAutoregressiveTransform):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._epsilon = 1e-1


class MAF(nn.Module):

    def __init__(self, n_layers, n_dim, n_latent_dim, n_hidden_features, dropout_probability=0.0, use_batch_norm=False, *args, **kwargs):
        super().__init__()
        self.n_layers = n_layers
        self.n_dim = n_dim
        self.n_latent_dim = n_latent_dim
        self.n_hidden_features = n_hidden_features
        self.use_batch_norm = use_batch_norm
        self.dropout_probability = dropout_probability

        transforms = []

        # # coupling instead of AR
        # for _ in range(self.n_layers):
        #     def create_net(in_features, out_features):
        #         return nets.ResidualNet(in_features, out_features, hidden_features=30, num_blocks=5)
        #     transforms.append(RandomPermutation(features=self.n_dim))
        #     transforms.append(AffineCouplingTransform(mask=torch.Tensor([1., 1., 1., 0., 0., 0.]), transform_net_create_fn=create_net))

        for _ in range(self.n_layers):
            transforms.append(ReversePermutation(features=self.n_dim))
            transforms.append(MaskedAffineAutoregressiveTransform(
                features=self.n_dim,
                hidden_features=self.n_hidden_features,
                use_batch_norm=self.use_batch_norm,
                dropout_probability=self.dropout_probability,
            ))

        transform = CompositeTransform(transforms)

        # Define a base distribution.
        base_distribution = StandardNormal(shape=[self.n_dim])

        # Combine into a flow. (For normalizing flows, see arXiv:1912.02762)
        self.flow = Flow(transform=transform, distribution=base_distribution)

    def forward(self, input):
        return [input]

    def sample(self, noise):
        samples, _ = self.flow._transform.inverse(noise)
        return samples

    def loss(self, *args, **kwargs):
        input = args[0]
        return {'loss': -self.flow.log_prob(inputs=input).mean()}
