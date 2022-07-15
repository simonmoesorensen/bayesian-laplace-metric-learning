from torchvision.datasets import FakeData
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split
from torchvision import transforms


class NoiseDataModule(LightningDataModule):
    def __init__(self, data_dir, batch_size, num_workers):
        super().__init__()

        self.name = "Noise"
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.n_classes = 10
        self.class_size = [5000] * 10

        self.transform = transforms.Compose([transforms.ToTensor()])

        self.df_train = None
        self.df_val = None
        self.df_test = None

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            self.df_train = FakeData(transform=self.transform, image_size=(3, 32, 32), size=50000)
            self.df_val = FakeData(transform=self.transform, image_size=(3, 32, 32), size=5000)

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.df_test = FakeData(transform=self.transform, image_size=(3, 32, 32), size=5000)

    def train_dataloader(self):
        return DataLoader(
            self.df_train,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
        )

    def val_dataloader(self):
        return DataLoader(
            self.df_val,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
        )

    def test_dataloader(self):
        return DataLoader(
            self.df_test,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
        )
