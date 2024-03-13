import argparse
from itertools import combinations
import numpy as np
import pandas as pd
import torch
import mlflow
from ray import tune, air
from ray.air.callbacks.mlflow import MLflowLoggerCallback
from ray.tune.schedulers import FIFOScheduler
from ray.tune.search.bayesopt import BayesOptSearch
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, RBF
from genhack.utils import COLS, anderson_darling, kendall_absolute_error

parser = argparse.ArgumentParser()
parser.add_argument('--year', type=str)
parser.add_argument('--cv', type=int, choices=[4, 5])
parser.add_argument('--kernel', type=str, choices=['RBF', 'Matern'])
parser.add_argument('--num_samples', type=int, default=500)
cli_args = parser.parse_args()

param_space = {
    "length_scale": tune.uniform(1., 20.),
    "constant_value": tune.uniform(0.01, 5.0),
}

if cli_args.kernel == 'Matern':
    param_space['nu'] = tune.uniform(1., 100.)


def objective(args):

    filename = '/Users/konstantin/projects/Flash/data/df_all.csv'
    df = pd.read_csv(filename)
    df['dates'] = pd.to_datetime(df['dates'])
    df = df.set_index('dates')[COLS]

    start_train_date = f'{cli_args.year}-01-01'
    end_train_date = f'{cli_args.year}-12-31'

    df = df[(df.index >= str(start_train_date)) & (df.index <= str(end_train_date))]
    df = torch.tensor(df.to_numpy())

    filename = '/Users/konstantin/projects/Flash/data/position.npy'
    position = np.load(filename).astype(np.float32)

    all_train_dims = [list(x) for x in list(combinations(range(6), cli_args.cv))]

    ad_means, kendalls = [], []

    for train_dims in all_train_dims:
        test_dims = [x for x in range(6) if x not in train_dims]

        sst_train = df[:, train_dims]
        positions_train = position.reshape(2, 6)[:, train_dims].reshape(-1, 2 * len(train_dims))
        sst_test = df[:, test_dims]
        positions_test = position.reshape(2, 6)[:, test_dims].reshape(-1, 2 * len(test_dims))

        y_samples = []

        noise = torch.randn((len(sst_train), len(test_dims)))

        for sst_daily, z in zip(sst_train, noise):
            if cli_args.kernel == 'RBF':
                kernel = ConstantKernel(constant_value=args['constant_value'], constant_value_bounds="fixed") * \
                         RBF(length_scale=args['length_scale'], length_scale_bounds="fixed")
            else:
                kernel = ConstantKernel(constant_value=args['constant_value'], constant_value_bounds="fixed") * \
                         Matern(length_scale=args['length_scale'], length_scale_bounds="fixed", nu=args['nu'])

            gpr = GaussianProcessRegressor(kernel=kernel)
            gpr.fit(positions_train.reshape(2, -1).T, sst_daily)
            y_mean, y_cov = gpr.predict(positions_test.reshape(2, -1).T, return_cov=True)

            y_cov += 1e-7 * np.eye(y_cov.shape[0])
            b = np.linalg.cholesky(y_cov)
            y_samples.append(y_mean + np.dot(b, z))

        y_samples = torch.tensor(y_samples)

        ad_ind, ad_mean = anderson_darling(y_samples, sst_test)
        kendall = kendall_absolute_error(y_samples, sst_test)

        ad_means.append(ad_mean)
        kendalls.append(kendall)

    return {'ad_mean': np.mean(ad_means), 'kendall': np.mean(kendalls)}


if __name__ == '__main__':
    print(f"MLFlow tracking URI: {mlflow.get_tracking_uri()}")
    print(f"MLFlow artifact URI: {mlflow.get_artifact_uri()}")

    opt = BayesOptSearch(metric="ad_mean", mode="min")

    scheduler = FIFOScheduler()

    tuner = tune.Tuner(
        objective,
        run_config=air.RunConfig(
            name="mlflow",
            callbacks=[MLflowLoggerCallback(
                tracking_uri=mlflow.get_tracking_uri(),
                experiment_name=f"{cli_args.kernel} {cli_args.year} CV{cli_args.cv}",
                save_artifact=True,
            )],
        ),
        param_space=param_space,
        tune_config=tune.TuneConfig(
            search_alg=opt,
            scheduler=scheduler,
            num_samples=cli_args.num_samples,
        ),
    )

    results = tuner.fit()
