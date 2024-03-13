import math

import mlflow.pytorch
import torch
from torch import optim
import pytorch_lightning as pl
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau

from genhack.models import GPR
from genhack.utils import calculate_ri, anderson_darling, log_hist2d, DEVICE, evaluate_model, kendall_absolute_error, log_gpr_contourplot


class Experiment(pl.LightningModule):

    def __init__(self, model, params, best_ad_mean_model_uri, best_kendall_model_uri, datamodule, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model
        self.params = params
        self.best_ad_mean_model_uri = best_ad_mean_model_uri
        self.best_kendall_model_uri = best_kendall_model_uri
        # a sort of anti-pattern, but we need metadata in the experiment
        self.datamodule = datamodule

        self.ri_true = None
        self.best_ad_mean = self.best_kendall = math.inf

    def lr_scheduler_step(self, scheduler, optimizer_idx, metric):
        super().lr_scheduler_step(scheduler, optimizer_idx, metric)
        mlflow.log_metric(f'lr_{optimizer_idx}', scheduler.optimizer.param_groups[0]['lr'], step=self.current_epoch)

    def configure_optimizers(self):
        if len(list(self.model.parameters())) > 0:
            adam = optim.Adam(self.model.parameters(), lr=self.params['learning_rate'])
            scheduler = ReduceLROnPlateau(adam, patience=self.params['patience'], factor=self.params['factor'], verbose=True)
            return {
                "optimizer": adam,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": self.params['lr_scheduler_metric'],
                },
            }

    def training_step(self, batch, batch_idx, optimizer_idx=None):
        result = self.model(batch)
        loss = self.model.loss(*result)
        self.log_dict({key: val.item() for key, val in loss.items()})
        return loss['loss']

    def validation_step(self, batch, batch_idx):

        # note that position is the same for all inputs in the batch
        X_val, position, time = batch[0], batch[1][0], batch[2]

        # sample
        X_val_pred = self.model.sample(torch.randn((len(X_val), self.model.n_latent_dim), device=DEVICE), position=position, time=time)

        # calculate Anderson-Darling
        ad_ind, ad_mean = anderson_darling(X_val, X_val_pred)

        # calculate Kendall explicitly to avoid the evaluation of ri_true at the end of every epoch
        kendall = kendall_absolute_error(X_val, X_val_pred)

        self.log_dict({'val_kendall': kendall, 'val_ad_mean': ad_mean})

        for i in range(X_val.shape[1]):
            self.log(f'val_ad_{i + 1}', ad_ind[i])

        # save best model
        if ad_mean < self.best_ad_mean:
            mlflow.pytorch.log_model(self.model, 'best_ad_mean')
            mlflow.log_metric('best_ad_mean_step', self.current_epoch)
            self.best_ad_mean = ad_mean
        if kendall < self.best_kendall:
            mlflow.pytorch.log_model(self.model, 'best_kendall')
            mlflow.log_metric('best_kendall_step', self.current_epoch)
            self.best_kendall = kendall

    def test_step(self, batch, batch_idx):

        print("Testing...")

        # note that position is the same for all inputs in the batch
        X_test, position, time = batch[0], batch[1][0], batch[2]

        log_hist2d('test_true', X_test)

        # log best AD
        best_model = mlflow.pytorch.load_model(self.best_ad_mean_model_uri)
        test_ba_kendall, test_ba_ad_ind, test_ba_ad_mean = \
            evaluate_model(best_model,
                           X_test=X_test,
                           position=position,
                           prefix='ba',
                           time=time,
                           n_latent_dim=self.model.n_latent_dim,
                           train_start_date=self.datamodule.train_start_date,
                           train_end_date=self.datamodule.train_end_date)

        # log best Kendall
        best_model = mlflow.pytorch.load_model(self.best_kendall_model_uri)
        test_bk_kendall, test_bk_ad_ind, test_bk_ad_mean = \
            evaluate_model(best_model,
                           X_test=X_test,
                           position=position,
                           prefix='bk',
                           time=time,
                           n_latent_dim=self.model.n_latent_dim,
                           train_start_date=self.datamodule.train_start_date,
                           train_end_date=self.datamodule.train_end_date)

        # if isinstance(best_model, GPR):
        #     for day in range(1):
        #         log_gpr_contourplot(best_model, day=day)

        return {
            f'test_ba_kendall': test_ba_kendall,
            f'test_ba_mean': test_ba_ad_mean,
            f'test_bk_kendall': test_bk_kendall,
            f'test_bk_mean': test_bk_ad_mean,
        }
