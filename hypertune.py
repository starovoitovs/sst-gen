from itertools import combinations
from hyperopt import hp

import mlflow
from ray import tune, air
from ray.tune.integration.mlflow import mlflow_mixin
from ray.tune.search import BasicVariantGenerator
import os

from ray.tune.search.hyperopt import HyperOptSearch

from genhack.utils import get_config
from run import run

train_dims = [list(x) for x in list(combinations(range(6), 5))]

param_space = {
    "['data_params']['train_dims']": tune.grid_search(train_dims),
    "kernel_type": hp.choice(['RBF', 'Matern']),
    "length_scale": hp.uniform([3., 10.]),
    "constant_value": hp.uniform([0.01, 2.0]),
    "nu": hp.uniform([2., 20.]),
    "mlflow": {
        "experiment_id": "12",
        "tracking_uri": mlflow.get_tracking_uri(),
    },
}


@mlflow_mixin
def objective(args):

    filename = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'configs/gpr.yaml')
    config = get_config(filename)

    for key, value in args.items():
        if key == 'mlflow' or key == "period":
            continue
        elif key.startswith('['):
            exec(f"config{key} = value")
        elif key == 'kernel_type':
            config['model_params']['kernel_params']['kernels'][1]['class_name'] = value[0]
            mlflow.log_param('kernel_type', value[0])
        elif key == 'nu':
            if config['model_params']['kernel_params']['kernels'][1]['class_name'] == 'Matern':
                config['model_params']['kernel_params']['kernels'][1]['kwargs']['nu'] = value
                mlflow.log_param('nu', value)
        elif key == 'constant_value':
            config['model_params']['kernel_params']['kernels'][0]['kwargs']['constant_value'] = value
            mlflow.log_param('constant_value', value)
        elif key == 'length_scale':
            config['model_params']['kernel_params']['kernels'][1]['kwargs']['length_scale'] = value
            mlflow.log_param('length_scale', value)
        else:
            raise ValueError(f"Unknown key {key}")

    config['data_params']['test_dims'] = [x for x in range(6) if x not in config['data_params']['train_dims']]
    return run(config, enable_progress_bar=False)


if __name__ == '__main__':

    print(f"MLFlow tracking URI: {mlflow.get_tracking_uri()}")
    print(f"MLFlow artifact URI: {mlflow.get_artifact_uri()}")

    hyperopt_search = HyperOptSearch(space, metric="mean_accuracy", mode="max")

    tuner = tune.Tuner(
        objective,
        run_config=air.RunConfig(name="mlflow"),
        param_space=param_space,
        tune_config=tune.TuneConfig(
            search_alg=BasicVariantGenerator(constant_grid_search=True, max_concurrent=4),
            num_samples=1,
        ),
    )

    results = tuner.fit()
