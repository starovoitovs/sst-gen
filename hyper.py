import mlflow
import numpy as np
from hyperopt import hp, fmin, tpe, space_eval
from ray.tune.integration.mlflow import mlflow_mixin

from genhack.utils import get_config
from run import run

space = {
    'model_params.n_blocks': hp.choice('n_blocks', np.arange(3, 5, 10)),
    'model_params.n_layers': hp.choice('n_layers', np.arange(5, 7, 10, 15)),
    'model_params.n_hidden_features': hp.choice('n_layers', np.arange(8, 16, 32, 64)),
    'model_params.dropout_probability': hp.choice('n_layers', np.arange(0., 0.25, 0.5, 0.75)),
    'model_params.use_batch_norm': hp.choice('n_layers', np.arange(True, False)),
    'experiment_params.learning_rate': hp.choice('n_layers', np.arange(0.001, 0.0001)),
}


@mlflow_mixin
def objective(args):

    config = get_config('configs/maf.yaml')

    for key, value in args.items():
        first, second = key.split('.')
        config[first][second] = value

    result = run(config)
    return float(result['test_ad_mean'])


"""
Check in detail, perhaps try simulated annealing:
https://www.kaggle.com/code/ilialar/hyperparameters-tunning-with-hyperopt/notebook
"""
if __name__ == '__main__':

    mlflow.set_experiment('Tuning MAF')

    # minimize the objective over the space
    best = fmin(objective, space, algo=tpe.suggest, max_evals=100)

    print(best)
    print(space_eval(space, best))
