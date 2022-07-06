from pathlib import Path
import torchvision.datasets as d
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

from data_modules.BaseDataModule import BaseDataModule


class MNISTDataModule(BaseDataModule):
    def __init__(self, data_dir, batch_size, num_workers):
        super().__init__(d.MNIST, data_dir, batch_size, num_workers)

        self.name = "MNIST"
        self.n_classes = 10

        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
