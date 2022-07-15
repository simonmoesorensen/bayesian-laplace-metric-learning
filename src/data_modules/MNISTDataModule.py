import torchvision.datasets as d
from torchvision import transforms

from src.data_modules.BaseDataModule import BaseDataModule


class MNISTDataModule(BaseDataModule):
    def __init__(self, data_dir, batch_size, num_workers, sampler=None, shuffle=False, pin_memory=True):
        super().__init__(d.MNIST, data_dir, batch_size, num_workers, sampler, shuffle, pin_memory)

        self.name = "MNIST"
        self.n_classes = 10

        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )

    def prepare_data(self):
        super().prepare_data()
        d.FashionMNIST(self.data_dir, train=False, download=True)

    def setup(self, val_split=0.2, shuffle=True):
        super().setup(val_split, shuffle)

        # Set FashionMNIST as OOD dataset
        ood_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                # Found using self._compute_mean_and_std()
                transforms.Normalize((0.2861), (0.3528)),
            ]
        )

        self.dataset_ood = d.FashionMNIST(
            self.data_dir, train=False, transform=ood_transforms
        )