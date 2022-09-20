import torchvision.datasets as d
from src.data_modules.BaseDataModule import BaseDataModule
from torchvision import transforms


class CIFAR10DataModule(BaseDataModule):
    def __init__(
        self,
        data_dir,
        batch_size,
        num_workers,
    ):
        super().__init__(
            d.CIFAR10, data_dir, batch_size, num_workers
        )

        self.name = "CIFAR"
        self.n_classes = 10

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )

    def setup(self, val_split=0.2, shuffle=True):
        super().setup(val_split, shuffle)

        self.dataset_ood = d.SVHN(self.data_dir, split="test", transform=self.transform)