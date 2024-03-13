import os
import numpy as np
import pandas as pd
import torch
from pytorch_lightning import LightningDataModule
from sklearn.model_selection import train_test_split
from torch import Tensor
from torch.utils.data import TensorDataset, default_collate, DataLoader

from genhack.utils import COLS, DEVICE

# automatically load all tensor on the device
# https://stackoverflow.com/questions/65932328/pytorch-while-loading-batched-data-using-dataloader-how-to-transfer-the-data-t
collate_fn = lambda x: tuple(y.to(DEVICE) for y in default_collate(x))


class PermutedTensorDataset(TensorDataset):

    def __init__(self, *tensors: Tensor, permute_coords):
        super().__init__(*tensors)
        self.permute_coords = permute_coords

    def __getitem__(self, index):
        sst, position, time = super().__getitem__(index)
        n_dim = sst.shape[0]

        if self.permute_coords:
            idx = torch.randperm(n_dim)
            lat, lon = position[:n_dim], position[n_dim:]
            sst = sst[idx]
            lat = lat[idx]
            lon = lon[idx]
            position = torch.cat([lat, lon])

        return sst, position, time


class StationsDataset(LightningDataModule):

    def __init__(self, batch_size, start_train_date, end_train_date, start_test_date, end_test_date, train_dims, test_dims, train_val_equal=False, val_split_size=0.2, train_val_shuffle=False, train_permute_coords=True, *args, **kwargs):
        super().__init__()
        self.batch_size = batch_size
        self.start_train_date = start_train_date
        self.end_train_date = end_train_date
        self.start_test_date = start_test_date
        self.end_test_date = end_test_date
        self.train_dims = train_dims
        self.test_dims = test_dims
        self.train_val_equal = train_val_equal
        self.val_split_size = val_split_size
        self.train_val_shuffle = train_val_shuffle
        self.train_permute_coords = train_permute_coords

        assert len(train_dims) >= len(test_dims), "train_dims must be larger than test_dims"

        filename = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../data/df_all.csv')
        self.df = pd.read_csv(filename)
        self.df['dates'] = pd.to_datetime(self.df['dates'])
        self.df = self.df.set_index('dates')[COLS]

        self.df_train_val = self.df[(self.df.index >= str(start_train_date)) & (self.df.index <= str(end_train_date))]
        self.df_test = self.df[(self.df.index >= str(start_test_date)) & (self.df.index <= str(end_test_date))]

        # t_min = min(self.df_train_val.index.min().timestamp(), self.df_test.index.min().timestamp())
        # t_max = max(self.df_train_val.index.max().timestamp(), self.df_test.index.max().timestamp())

        t_min = 368150400.0
        t_max = 1483142400.0

        time_train_val = torch.tensor((self.df_train_val.index.astype(int) / 10 ** 9 - t_min) / (t_max - t_min)).float()
        time_test = torch.tensor((self.df_test.index.astype(int) / 10 ** 9 - t_min) / (t_max - t_min)).float()

        date_train_val = self.df_train_val.index
        date_test = self.df_test.index

        X_train_val = torch.tensor(self.df_train_val.to_numpy().astype(np.float32), device=DEVICE)
        X_test = torch.tensor(self.df_test.to_numpy().astype(np.float32), device=DEVICE)

        filename = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../data/position.npy')
        position = np.load(filename).astype(np.float32)
        positions_test = torch.tensor(position).repeat(len(X_test)).reshape(len(X_test), -1)
        positions_train_val = torch.tensor(position).repeat(len(X_train_val)).reshape(len(X_train_val), -1)

        if train_val_equal:
            X_train, date_train, time_train, positions_train = X_train_val, date_train_val, time_train_val, positions_train_val
            X_val, date_val, time_val, positions_val = X_train_val, date_train_val, time_train_val, positions_train_val
        else:
            X_train, X_val, date_train, date_val, time_train, time_val, positions_train, positions_val = \
                train_test_split(X_train_val, date_train_val, time_train_val, positions_train_val, test_size=val_split_size, shuffle=train_val_shuffle)

        X_train = X_train[:, train_dims]
        positions_train = positions_train.reshape(-1, 2, 6)[:, :, train_dims].reshape(-1, 2 * len(train_dims))

        X_val = X_val[:, test_dims]
        positions_val = positions_val.reshape(-1, 2, 6)[:, :, test_dims].reshape(-1, 2 * len(test_dims))

        X_test = X_test[:, test_dims]
        positions_test = positions_test.reshape(-1, 2, 6)[:, :, test_dims].reshape(-1, 2 * len(test_dims))

        self.train_start_date = date_train.min()
        self.train_end_date = date_train.max()
        self.val_start_date = date_val.min()
        self.val_end_date = date_val.max()
        self.test_start_date = date_test.min()
        self.test_end_date = date_test.max()

        self.train_dataset = PermutedTensorDataset(X_train, positions_train, time_train, permute_coords=self.train_permute_coords)
        self.val_dataset = TensorDataset(X_val, positions_val, time_val)
        self.test_dataset = TensorDataset(X_test, positions_test, time_test)

        self.t_val_min = time_val.min()
        self.t_val_max = time_val.max()
        self.t_test_min = time_test.min()
        self.t_test_max = time_test.max()

    def train_dataloader(self):
        return DataLoader(dataset=self.train_dataset, collate_fn=collate_fn, drop_last=True, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(dataset=self.val_dataset, collate_fn=collate_fn, batch_size=100000)

    def test_dataloader(self):
        return DataLoader(dataset=self.test_dataset, collate_fn=collate_fn, batch_size=100000)
