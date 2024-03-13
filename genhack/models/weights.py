import torch
from torch import nn

from genhack.utils import DEVICE


class PowerLawWeights(nn.Module):

    def __init__(self, a=0.9, b=0.1, c=1., *args, **kwargs):
        """(a * t + b) ** c"""
        super().__init__()
        self.a = a
        self.b = b
        self.c = c

    def forward(self, input):
        return (self.a * input + self.b) ** self.c


class LearnableWeights(nn.Module):

    def __init__(self, pts=64, n_hidden_units=100, *args, **kwargs):
        """
        Learns non-linear function [0,1]->[0,inf] which integrates to one.

        Parameters
        ----------
        pts : int
            Integration points for calculation of the normalizing constant, so the weights integrate to one
        n_hidden_units
            Number of hidden units in the weight function
        """
        super().__init__()
        self.pts = pts
        self.n_hidden_units = n_hidden_units
        self.model = nn.Sequential(
            nn.Linear(1, n_hidden_units),
            nn.LeakyReLU(),
            nn.Linear(n_hidden_units, 1),
            nn.Sigmoid(),
        )

    def forward(self, input):
        dt = 1 / self.pts
        normalize = self.model(torch.arange(0, 1, dt, device=DEVICE)[:, None]).sum() * dt
        return self.model(input[:, None]).reshape(-1) / normalize
