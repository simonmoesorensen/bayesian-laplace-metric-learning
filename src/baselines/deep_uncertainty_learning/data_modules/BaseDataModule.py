from pytorch_lightning import LightningDataModule
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split


class BaseDataModule(LightningDataModule):
    def __init__(self, dataset_cls, data_dir, batch_size, num_workers):
        super().__init__()

        self.cls = dataset_cls
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.eval_batch_size = batch_size
        self.num_workers = num_workers

        self.df_train = None
        self.df_val = None
        self.df_test = None

    def prepare_data(self):
        self.cls(self.data_dir, train=True, download=True)
        self.cls(self.data_dir, train=False, download=True)

    def setup(self, stage=None, val_split=0.2):
        assert self.transform, "transform must be set before setup()"

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            dataset_full = self.cls(self.data_dir, train=True, transform=self.transform)
            size = len(dataset_full)

            def get_split_size(frac):
                return int(size // (1 / frac))

            self.dataset_train, self.dataset_val = random_split(
                dataset_full, [get_split_size(1 - val_split), get_split_size(val_split)]
            )

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.dataset_test = self.cls(
                self.data_dir, train=False, transform=self.transform
            )

    def train_dataloader(self):
        return DataLoader(
            self.dataset_train,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
        )

    def val_dataloader(self):
        return DataLoader(
            self.dataset_val,
            num_workers=self.num_workers,
            batch_size=self.eval_batch_size,
        )

    def test_dataloader(self):
        return DataLoader(
            self.dataset_test,
            num_workers=self.num_workers,
            batch_size=self.eval_batch_size,
        )
