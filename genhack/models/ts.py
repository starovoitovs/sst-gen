import torch
from sklearn.linear_model import LinearRegression
from torch import nn


class TrendModel(nn.Module):

    def __init__(self, data, trend_factor=1.) -> None:
        super().__init__()
        self.data = data
        self.trend_factor = trend_factor

        self.coef = None
        self.intercept = None

    def forward(self, time):
        return self.intercept[None, :] + self.trend_factor * self.coef[None, :] * time[:, None]

    def fit(self):
        lr_data = self.data.resample('Y').mean()
        lr = LinearRegression()
        xs = torch.linspace(0, 1, len(lr_data))[:, None]
        reg = lr.fit(xs, lr_data)

        self.coef = nn.Parameter(torch.tensor(reg.coef_.reshape(-1)), requires_grad=False)
        self.intercept = nn.Parameter(torch.tensor(reg.intercept_.reshape(-1)), requires_grad=False)
