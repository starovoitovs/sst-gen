
# GenHack2 - Hackathon for Generative modeling : Simulation of global warming Sea Surface Temperatures ([website](https://www.polytechnique.edu/en/education/academic-and-research-departments/applied-mathematics-department-depmap/student-event/genhack-2-hackathon-generative-modelling))
<img src="https://www.polytechnique.edu/sites/default/files/styles/contenu_detail/public/content/pages/images/2022-10/GenHack%20Challenge%20%28Banni%C3%A8re%20%28paysage%29%29%20%281250%20%C3%97%20350%20px%29_0.png?itok=K1AwTb_0">

This project is a joint work with @saschagaudlitz and @ziebelle.

## Model development

> See `models/maf.py` for a working example.

You need to subclass torch's `nn.Module` and implement `forward`, `sample` and `loss` methods.

In the end, import contents of your module in the `models/__init__.py` file and add your model to the `models` dictionary, for example:

    from .maf import *
    ...
    
    models = {
        ...
        'MAF': MAF,
    }

# MLFlow

We use MLFlow for model tracking.

    pip install mlflow

Then start the MLFlow from the root directory of the project (better specify absolute path due to `ray`):

    mlflow server --backend-store-uri=sqlite:///mlflow.db --default-artifact-root=/Users/konstantin/projects/Flash/mlruns

Then you can access the server under `127.0.0.1:5000`.

Note that in `mlflow` you can run an *experiment* that has several *runs*. For example, for hyperparameter tuning or cross-validation, you would run one experiment with multiple runs.

# Running models

## From command line

You can run a model from the root directory of the project by specifying the name of the class and the config argument as follows:

    MLFLOW_TRACKING_URI=sqlite:///mlflow.db python3 run.py --config=configs/vae.yaml

## From notebook

You can also run the model from the notebook, see `train.ipynb` for the example. In the notebook you can monitor 2d-marginals graphically during training.

## Hyperparameter tuning

You need to install `hyperopt` and `ray` for hyperparameter tuning. You also need to specify the `MLFLOW_TRACKING_URI` environment variable. Note that you always need to specift the absolute path to the database file.

    MLFLOW_TRACKING_URI=sqlite:////Users/konstantin/projects/Flash/mlflow.db pip install hyperopt ray

See `hypertune.py` for example script.

## Testing model

Testing metrics and plots are recorded after the training run. Alternatively, you can run testing separately (for example, with another seed):

    python3 run.py --config=configs/maf.yaml --mode=test --run_id=1af75f70444d43999157bbf6c20bccda
