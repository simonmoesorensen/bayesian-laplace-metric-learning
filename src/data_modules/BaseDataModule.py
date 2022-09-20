import numpy as np
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Subset, random_split
from tqdm import tqdm


class BaseDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir,
        batch_size,
        num_workers,
    ):
        super().__init__()

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.dataset_train = None
        self.dataset_val = None
        self.dataset_test = None
        self.dataset_ood = None

    def train_dataloader(self):
        return DataLoader(
            self.dataset_train,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            pin_memory=True,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.dataset_val,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            pin_memory=True,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.dataset_test,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            pin_memory=True,
            shuffle=False,
        )

    def ood_dataloader(self):
        return DataLoader(
            self.dataset_ood,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            pin_memory=True,
            shuffle=False,
        )