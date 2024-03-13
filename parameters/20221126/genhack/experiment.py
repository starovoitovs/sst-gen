import math
import mlflow.pytorch
import torch
from torch import optim
import pytorch_lightning as pl

from genhack.utils import calculate_ri, anderson_darling, log_test_metrics, log_hist2d, DEVICE


class Experiment(pl.LightningModule):

    def __init__(self, model, params, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model
        self.params = params
        self.ri_true = None

        self.best_ad_mean = self.best_kendall = math.inf
        self.best_ad_mean_model_uri = self.best_kendall_model_uri = None

    def configure_optimizers(self):
        if len(list(self.model.parameters())) > 0:
            adam = optim.Adam(self.model.parameters(), lr=self.params['learning_rate'])
            return adam
            # scheduler = ReduceLROnPlateau(adam, mode='min', factor=0.1)
            # return [adam], [scheduler]

    def training_step(self, batch, batch_idx):
        result = self.model(batch[0])
        loss = self.model.loss(*result)
        self.log_dict({key: val.item() for key, val in loss.items()})
        return loss['loss']

    def validation_step(self, batch, batch_idx):

        X_val = batch[0]

        # calculate Kendall ri for the validation set only once, because this operation takes a couple of seconds
        if self.ri_true is None:
            self.ri_true = calculate_ri(X_val)

        X_val_pred = self.model.sample(torch.randn((len(X_val), self.model.n_latent_dim), device=DEVICE))
        ad_ind, ad_mean = anderson_darling(X_val, X_val_pred)

        # calculate Kendall explicitly to avoid the evaluation of ri_true at the end of every epoch
        ri_pred = calculate_ri(X_val_pred)
        kendall = torch.abs(ri_pred - self.ri_true).mean()

        self.log_dict({'val_kendall': kendall, 'val_ad_mean': ad_mean})

        for i in range(self.model.n_dim):
            self.log(f'val_ad_{i + 1}', ad_ind[i])

        # save best model
        if ad_mean < self.best_ad_mean:
            model_info = mlflow.pytorch.log_model(self.model, 'best_ad_mean')
            self.best_ad_mean = ad_mean
            self.best_ad_mean_model_uri = model_info.model_uri
        if kendall < self.best_kendall:
            model_info = mlflow.pytorch.log_model(self.model, 'best_kendall')
            self.best_kendall = kendall
            self.best_kendall_model_uri = model_info.model_uri

    def test_step(self, batch, batch_idx):

        X_test = batch[0]
        log_hist2d('test_true', X_test)

        best_model = mlflow.pytorch.load_model(self.best_ad_mean_model_uri)
        X_test_pred = best_model.sample(torch.randn((len(X_test), self.model.n_latent_dim), device=DEVICE))
        test_ba_kendall, test_ba_ad_mean = log_test_metrics(X_test, X_test_pred, 'ba')
        log_hist2d(f'test_ba_pred', X_test_pred, X_test)

        best_model = mlflow.pytorch.load_model(self.best_kendall_model_uri)
        X_test_pred = best_model.sample(torch.randn((len(X_test), self.model.n_latent_dim), device=DEVICE))
        test_bk_kendall, test_bk_ad_mean = log_test_metrics(X_test, X_test_pred, 'bk')
        log_hist2d(f'test_bk_pred', X_test_pred, X_test)

        return {
            f'test_ba_kendall': test_ba_kendall,
            f'test_ba_mean': test_ba_ad_mean,
            f'test_bk_kendall': test_bk_kendall,
            f'test_bk_mean': test_bk_ad_mean,
        }
