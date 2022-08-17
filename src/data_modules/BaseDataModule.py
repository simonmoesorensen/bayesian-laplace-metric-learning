from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split, Subset
import numpy as np
from tqdm import tqdm
import torch


class BaseDataModule(LightningDataModule):
    def __init__(
        self,
        dataset_cls,
        data_dir,
        batch_size,
        num_workers,
        sampler=None,
        shuffle=False,
        pin_memory=True,
    ):
        super().__init__()

        self.cls = dataset_cls
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.df_train = None
        self.df_val = None
        self.df_test = None

        # Dataloader args
        self.sampler = sampler
        self.shuffle = shuffle
        self.pin_memory = pin_memory

    def prepare_data(self):
        self.cls(self.data_dir, train=True, download=True)
        self.cls(self.data_dir, train=False, download=True)

    def setup(self, val_split=0.2, shuffle=True):
        assert self.transform, "transform must be set before setup()"

        # Assign train/val datasets for use in dataloaders
        dataset_full = self.cls(self.data_dir, train=True, transform=self.transform)
        size = len(dataset_full)

        def get_split_size(frac):
            return np.round(size // (1 / frac)).astype(int)

        n_train = get_split_size(1 - val_split)
        n_val = get_split_size(val_split)

        # Ensure that the splits cover the whole dataset
        n_train += size - n_train - n_val

        if shuffle:
            # Overlapping classes allowed
            self.dataset_train, self.dataset_val = random_split(
                dataset_full, [n_train, n_val]
            )
        else:
            # Overlapping classes not allowed (zero-shot learning)
            self.dataset_train = Subset(dataset_full, range(0, n_train))
            self.dataset_val = Subset(dataset_full, range(n_train, n_train + n_val))

        # Assign test dataset for use in dataloader(s)
        self.dataset_test = self.cls(
            self.data_dir, train=False, transform=self.transform
        )

    def train_dataloader(self):
        return DataLoader(
            self.dataset_train,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            pin_memory=self.pin_memory,
            sampler=self.sampler,
            shuffle=False if self.sampler else self.shuffle,
        )

    def val_dataloader(self):
        return DataLoader(
            self.dataset_val,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            pin_memory=self.pin_memory,
            shuffle=False if self.sampler else self.shuffle,
        )

    def test_dataloader(self):
        return DataLoader(
            self.dataset_test,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            pin_memory=self.pin_memory,
            shuffle=False if self.sampler else self.shuffle,
        )

    def ood_dataloader(self):
        return DataLoader(
            self.dataset_ood,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            pin_memory=self.pin_memory,
            shuffle=False if self.sampler else self.shuffle,
        )

    def _compute_mean_and_std(self, dataset):
        dataloader = DataLoader(
            dataset, num_workers=self.num_workers, batch_size=self.batch_size
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        mean = []
        std = []
        for img, _ in tqdm(dataloader):
            img = img.to(device)

            mean.append(img.mean([0, 2, 3]))
            std.append(img.std([0, 2, 3]))

        mean = torch.stack(mean).mean(0)
        std = torch.stack(std).mean(0)

        print(f"mean: {mean}")
        print(f"std: {std}")
        return mean, std
