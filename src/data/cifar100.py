import torch
from torch.utils.data import Subset
from torchvision.datasets import CIFAR100
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split
from torchvision import transforms


class CIFAR100DataModule(LightningDataModule):
    def __init__(self, data_dir, batch_size, num_workers):
        super().__init__()

        self.name = "CIFAR-100"
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.n_classes = 10

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.5071, 0.4867, 0.4408),
                    (0.2675, 0.2565, 0.2761),
                ),
            ]
        )

        self.df_train = None
        self.df_val = None
        self.df_test = None

    def prepare_data(self):
        # download
        CIFAR100(self.data_dir, train=True, download=True)
        # CIFAR100(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            train = CIFAR100(self.data_dir, train=True, transform=self.transform)
            subset_classes = range(self.n_classes)
            mask = torch.tensor([train[i][1] in subset_classes for i in range(len(train))])
            indices = torch.arange(len(train))[mask]
            train_10classes = Subset(train, indices)
            self.df_test = train_10classes

    def train_dataloader(self):
        raise NotImplementedError

    def val_dataloader(self):
        raise NotImplementedError

    def test_dataloader(self):
        return DataLoader(
            self.df_test,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            shuffle=False,
        )
