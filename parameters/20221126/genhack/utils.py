import argparse
import mlflow
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import torch
import yaml
from PIL import Image


COLS = ['s1', 's2', 's3', 's4', 's5', 's6']
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_config(filename):

    with open(filename, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    return config


def calculate_ri(X):
    """Calculate order statistics for R_{i,n_test}.
    """
    n_test = X.shape[0]
    # we can ignore j = i, since it's zero anyway
    return torch.sort(torch.sum(torch.prod(X[:, None, :] < X[None, :, :], dim=2), dim=1) / (n_test - 1), dim=0).values


def kendall_absolute_error(X_true, X_pred):
    """Kendall absolute error metric
    """
    return torch.abs(calculate_ri(X_true) - calculate_ri(X_pred)).mean()


def anderson_darling(X_true, X_pred):
    """Anderson darling metric
    """
    assert X_true.shape == X_pred.shape
    n_test = X_true.shape[0]

    X_pred_sorted = torch.sort(X_pred, dim=0).values
    u = (torch.sum(X_true[None, :] <= X_pred_sorted[:, None], dim=1) + 1) / (n_test + 2)

    ad_ind = -n_test - torch.sum((2 * torch.arange(1, n_test + 1, device=DEVICE) - 1).reshape(-1, 1) * (torch.log(u) + torch.log(1 - torch.flip(u, dims=(0,)))), dim=0) / n_test
    ad_mean = ad_ind.mean()

    return ad_ind, ad_mean


def plot_hist2d(X, X_true=None):
    """Plot 2d histogram of the data.
    X_true is optional, if provided we draw two 1d-histograms on the diagonal and metrics in plot titles.
    """
    kendall = ad_ind = None

    if X_true is not None:
        kendall = kendall_absolute_error(X_true, X)
        ad_ind, ad_mean = anderson_darling(X_true, X)

    n_dim = X.shape[1]

    fig, ax = plt.subplots(nrows=n_dim, ncols=n_dim, figsize=(3 * n_dim, 3 * n_dim - 2), constrained_layout=True)

    for i in range(n_dim):
        for j in range(n_dim):
            if i < j:
                ax[i][j].hist2d(X[:, i].detach().numpy(), X[:, j].detach().numpy(), bins=50, range=[[-7, 7], [-7, 7]])
            if i > j:
                # delete axes below diagonal
                fig.delaxes(ax[i][j])

    for i in range(n_dim):
        data = pd.DataFrame(torch.vstack([X_true[:, i], X[:, i]]).T.detach().numpy()) if X_true is not None else X[:, i].detach().numpy()
        sns.kdeplot(data=data, ax=ax[i][i])
        if ad_ind is not None:
            ax[i][i].set_title(f"AD = {ad_ind[i]:.4f}")
            ax[i][i].get_legend().remove()

    if kendall is not None:
        ax[0][1].set_title(f"Kendall = {kendall:.6f}")

    return fig


def log_test_metrics(X_true, X_pred, prefix):
    """Logs Kendall absolute error and Anderson-Darling distances in MLFlow.
    Note that this function should be run in the MLFlow context.
    """
    n_dim = X_true.shape[1]

    kendall = kendall_absolute_error(X_true, X_pred)
    ad_ind, ad_mean = anderson_darling(X_true, X_pred)

    mlflow.log_metric(f'test_{prefix}_kendall', kendall)
    mlflow.log_metric(f'test_{prefix}_ad_mean', ad_mean)

    for i in range(n_dim):
        mlflow.log_metric(f'test_{prefix}_ad_{i + 1}', ad_ind[i])

    return kendall, ad_mean


def log_hist2d(label, X, X_true=None):
    """Log 2d-histograms in MLFlow.
    """
    fname = f'hist2d_{label}.png'
    fig = plot_hist2d(X, X_true)
    fig.suptitle(fname, fontsize=20)
    fig.canvas.draw()
    image = Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
    mlflow.log_image(image, fname)
