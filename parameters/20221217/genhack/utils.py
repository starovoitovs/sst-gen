from itertools import permutations

import matplotlib
import mlflow
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import torch
import yaml
from PIL import Image
import collections.abc

from tqdm import tqdm

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

    fig, ax = plt.subplots(nrows=n_dim, ncols=n_dim, figsize=(3 * n_dim, max(3 * n_dim - 2, 3)), constrained_layout=True)

    for i in range(n_dim):
        for j in range(n_dim):
            if i < j:
                ax[i][j].hist2d(X[:, i].detach().cpu().numpy(), X[:, j].detach().cpu().numpy(), bins=50, range=[[-7, 7], [-7, 7]])
            if i > j:
                # delete axes below diagonal
                fig.delaxes(ax[i][j])

    for i in range(n_dim):
        data = pd.DataFrame(torch.vstack([X_true[:, i], X[:, i]]).T.detach().cpu().numpy()) if X_true is not None else X[:, i].detach().cpu().numpy()
        subplt = ax if n_dim == 1 else ax[i][i]
        sns.kdeplot(data=data, ax=subplt)
        if ad_ind is not None:
            subplt.set_title(f"AD = {ad_ind[i]:.4f}")
            subplt.get_legend().remove()

    if kendall is not None and n_dim > 1:
        ax[0][1].set_title(f"Kendall = {kendall:.6f}")

    return fig


def evaluate_model(model, prefix, X_test, position, time, n_latent_dim, train_start_date, train_end_date):

    n_test_dim = len(position) // 2

    kendall_arr, ad_ind_arr, ad_mean_arr = [], [], []
    X_test_pred = None

    N_TEST_SAMPLES = 10

    for _ in range(N_TEST_SAMPLES):

        # make predictions
        noise = torch.randn((len(X_test), n_latent_dim), device=DEVICE)
        X_test_pred = model.sample(noise, position=position, time=time)
        # cut-off dummy positions
        X_test_pred = X_test_pred[:, :n_test_dim]

        # calculate metrics
        kendall = kendall_absolute_error(X_test, X_test_pred)
        ad_ind, ad_mean = anderson_darling(X_test, X_test_pred)

        kendall_arr.append(kendall)
        ad_ind_arr.append(ad_ind[None, :])
        ad_mean_arr.append(ad_mean)

    test_kendall = torch.tensor(kendall_arr).mean(dim=0)
    test_ad_ind = torch.cat(ad_ind_arr, dim=0).mean(dim=0)
    test_ad_mean = torch.tensor(ad_mean_arr).mean(dim=0)

    # logs Kendall absolute error and Anderson-Darling distances in MLFlow.

    mlflow.log_metric(f'test_{prefix}_kendall', test_kendall)
    mlflow.log_metric(f'test_{prefix}_ad_mean', test_ad_mean)

    for i in range(test_ad_ind.shape[0]):
        mlflow.log_metric(f'test_{prefix}_ad_{i + 1}', test_ad_ind[i])

    # log the histogram

    log_hist2d(f'test_{prefix}_pred', X_test_pred, X_test)

    if hasattr(model, 'weights'):
        date_range = pd.date_range(train_start_date, train_end_date, freq='M')
        weights = model.weights(torch.linspace(0, 1, len(date_range))[:, None]).detach().numpy()
        log_weights(f'test_{prefix}_weights', date_range, weights)

    return test_kendall, test_ad_ind, test_ad_mean


def log_hist2d(label, X, X_true=None):
    """Log 2d-histograms in MLFlow.
    """
    fname = f'figures/hist2d_{label}.png'
    fig = plot_hist2d(X, X_true)
    fig.suptitle(fname, fontsize=20)
    fig.canvas.draw()
    image = Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
    mlflow.log_image(image, fname)


def log_weights(label, date_range, weights):
    fname = f'figures/{label}.png'
    fig, ax = plt.subplots(figsize=(5, 4), constrained_layout=True)
    ax.plot(date_range, weights)
    fig.canvas.draw()
    image = Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
    mlflow.log_image(image, fname)


def plot_gpr_contourplot(model, day=0, gridsize=48):
    """Plot contour plot of the GPR model.
    """
    fig, ax = plt.subplots(figsize=(15, 5), ncols=3)

    lon = torch.linspace(-16, 16, gridsize)
    lat = torch.linspace(-16, 16, gridsize)
    LON, LAT = torch.meshgrid(lon, lat)
    posgrid = torch.cat([LAT.reshape(-1), LON.reshape(-1)])
    noise = torch.randn((1, gridsize ** 2))

    mean, std = model.gprs[day].predict(posgrid.reshape(2, -1).T, return_std=True)
    mean = mean.reshape(gridsize, gridsize)
    std = std.reshape(gridsize, gridsize)

    ax[0].contourf(LON, LAT, mean, 200)
    ax[0].set_title('mean')

    ax[1].contourf(LON, LAT, std, 200)
    ax[1].set_title('std')

    samples = model.sample(noise, posgrid, start_day=day).reshape(gridsize, gridsize)
    ax[2].contourf(LON, LAT, samples, 200)
    ax[2].set_title('sample')

    cmap = plt.get_cmap('viridis')
    norm = matplotlib.colors.Normalize(vmin=mean.min(), vmax=mean.max())

    for (lat, lon), sst in zip(model.gprs[day].X_train_, model.gprs[day].y_train_):
        ax[2].scatter(lon, lat, color=cmap(norm(sst)), s=100, edgecolor='white', linewidth=1)

    return fig


def log_gpr_contourplot(model, day=0):
    fname = f"figures/gpr_contourplot_{day}.png"
    fig = plot_gpr_contourplot(model, day)
    fig.canvas.draw()
    image = Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
    mlflow.log_image(image, fname)
