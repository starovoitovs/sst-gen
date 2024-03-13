#############################################################################
# YOUR GENERATIVE MODEL
# ---------------------
# Should be implemented in the 'generative_model' function
# !! *DO NOT MODIFY THE NAME OF THE FUNCTION* !!
#
# You can store your parameters in any format you want (npy, h5, json, yaml, ...)
# <!> *SAVE YOUR PARAMETERS IN THE parameters/ DICRECTORY* <!>
#
# See below an example of a generative model
# G_\theta(Z) = np.max(0, \theta.Z)
############################################################################

import numpy as np
import os

import sys

sys.path.append('/parameters/20221217')

import mlflow
import torch


# <!> DO NOT ADD ANY OTHER ARGUMENTS <!>
def generative_model(noise, position):
    """
    Generative model

    Parameters
    ----------
    noise : ndarray with shape (n_samples, n_dim)
        input noise of the generative model
    """
    # See below an example
    # ---------------------
    latent_variable = torch.Tensor(noise[:, :7])
    model = mlflow.pytorch.load_model('parameters/20221217/best_ad_mean')
    samples = model.sample(latent_variable, position).detach().numpy()

    return samples
