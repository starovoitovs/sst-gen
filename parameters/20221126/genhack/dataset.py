from pytorch_lightning import LightningDataModule
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import os
from genhack.utils import COLS
from torch.utils.data.dataloader import default_collate


# https://stackoverflow.com/questions/65932328/pytorch-while-loading-batched-data-using-dataloader-how-to-transfer-the-data-t
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
collate_fn = lambda x: tuple(x_.to(device) for x_ in default_collate(x))

class StationsDataset(LightningDataModule):

    def __init__(self, batch_size, val_split_size=0.2, *args, **kwargs):
        super().__init__()
        self.batch_size = batch_size
        self.val_split_size = val_split_size

        filename = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../data/df_train.csv')
        X = pd.read_csv(filename)[COLS].to_numpy()
        X_train, X_val = train_test_split(X, test_size=val_split_size)

        filename = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../data/df_test.csv')
        X_test = pd.read_csv(filename)[COLS].to_numpy()

        self.train_dataset = TensorDataset(torch.tensor(X_train.astype(np.float32)))
        self.val_dataset = TensorDataset(torch.tensor(X_val.astype(np.float32)))
        self.test_dataset = TensorDataset(torch.tensor(X_test.astype(np.float32)))

    def train_dataloader(self):
        return DataLoader(dataset=self.train_dataset, collate_fn=collate_fn, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(dataset=self.val_dataset, collate_fn=collate_fn, batch_size=100000)

    def test_dataloader(self):
        return DataLoader(dataset=self.test_dataset, collate_fn=collate_fn, batch_size=100000)
