import argparse

import mlflow
import torch
from mlflow.entities import ViewType
from pytorch_lightning import Trainer
from pytorch_lightning.utilities.seed import seed_everything
from sklearn.linear_model import LinearRegression
from torch import nn
from tqdm import tqdm

from genhack.datasets import StationsDataset
from genhack.experiments import Experiment, experiments
from genhack.models import models, ts_models, weights_models
from genhack.utils import get_config, DEVICE


def run(config, mode='train', enable_progress_bar=True, callbacks=None):

    seed_everything(config['experiment_params']['manual_seed'], True)
    datamodule = StationsDataset(**config['data_params'])

    # initialize ts model
    ts_model = None
    if 'ts_model_params' in config['model_params']:
        ts_model_params = config['model_params']['ts_model_params']
        ts_model = ts_models[ts_model_params['model_name']](datamodule.df_train_val, **ts_model_params['kwargs'])

    # initialize weights model
    weights_model = None
    if 'weights_model_params' in config['model_params']:
        weights_model_params = config['model_params']['weights_model_params']
        weights_model = weights_models[weights_model_params['model_name']](**weights_model_params['kwargs'])

    n_dim = len(config['data_params']['train_dims'])
    n_latent_dim = len(config['data_params']['test_dims']) + 1
    model = models[config['model_params']['model_name']](**config['model_params'], datamodule=datamodule, ts_model=ts_model, weights_model=weights_model, n_dim=n_dim, n_latent_dim=n_latent_dim)

    # initialize experiment

    active_run = mlflow.active_run()
    best_ad_mean_model_uri = f'runs:/{active_run.info.run_id}/best_ad_mean'
    best_kendall_model_uri = f'runs:/{active_run.info.run_id}/best_kendall'

    experiment_class = config['experiment_params'].get('experiment_class', None)
    cls = experiments.get(experiment_class, Experiment)

    # @todo here is potential problem, as actual training dates, and training dates from the config might deviate
    experiment = cls(model, config.get('experiment_params', None),
                     best_ad_mean_model_uri=best_ad_mean_model_uri,
                     best_kendall_model_uri=best_kendall_model_uri,
                     datamodule=datamodule)

    # training

    if mode == 'train':

        mlflow.log_dict(config, 'config.yaml')

        mlflow.log_param('train_start_date', datamodule.train_start_date)
        mlflow.log_param('train_end_date', datamodule.train_end_date)
        mlflow.log_param('val_start_date', datamodule.val_start_date)
        mlflow.log_param('val_end_date', datamodule.val_end_date)
        mlflow.log_param('test_start_date', datamodule.test_start_date)
        mlflow.log_param('test_end_date', datamodule.test_end_date)

        # train generative model

        if callbacks is None:
            callbacks = []

        # num_sanity_val_steps = 0 is important, otherwise resets best metrics from inf to arbitrary value!
        num_sanity_val_steps = config['trainer_params'].get('num_sanity_val_steps', 0)
        config['trainer_params'] = {k: v for k, v in config['trainer_params'].items() if k != 'num_sanity_val_steps'}
        trainer = Trainer(callbacks=callbacks, enable_progress_bar=enable_progress_bar, num_sanity_val_steps=num_sanity_val_steps, **config['trainer_params'])

        mlflow.pytorch.autolog(log_models=False)

        # if the param name is longer than 250 chars, there is going to be an error
        for name in 'model_params', 'experiment_params', 'data_params', 'trainer_params':
            if name in config:
                for k, v in config[name].items():
                    if not isinstance(v, dict):
                        mlflow.log_param(k, v)

        trainer.fit(experiment, datamodule=datamodule)

    result = experiment.test_step([x.to(DEVICE) for x in datamodule.test_dataset[:]], 0)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, choices=['train', 'test'], default='train')
    parser.add_argument('--config', dest="filename", metavar='FILE')
    parser.add_argument('--experiment_id', type=str, default="0")
    parser.add_argument('--run_id', type=str, default=None)
    parser.add_argument('--all', action='store_true')
    args = parser.parse_args()

    assert not (args.mode == 'test' and args.filename is not None), "--config is invalid in `test` mode, it's retrieved from mlflow."
    assert not (args.mode == 'train' and args.run_id is not None), "--run_id is invalid in `train` mode, it's generated automatically."

    mlflow.set_experiment(experiment_id=args.experiment_id)

    if args.mode == 'test' and args.all:
        finished_runs = mlflow.search_runs(run_view_type=ViewType.ACTIVE_ONLY, filter_string="attribute.status = 'FINISHED'")
        for run_id in tqdm(finished_runs['run_id'].values):
            with mlflow.start_run(run_id=run_id) as active_run:
                filename = args.filename if args.mode == 'train' else mlflow.get_artifact_uri(artifact_path='config.yaml')
                config = get_config(filename)
                run(config, mode=args.mode)
    else:
        with mlflow.start_run(run_id=args.run_id) as active_run:
            filename = args.filename if args.mode == 'train' else mlflow.get_artifact_uri(artifact_path='config.yaml')
            config = get_config(filename)
            run(config, mode=args.mode)
