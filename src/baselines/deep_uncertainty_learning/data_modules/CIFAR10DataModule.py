import torchvision.datasets as d
from torchvision import transforms

from data_modules.BaseDataModule import BaseDataModule


class CIFAR10DataModule(BaseDataModule):
    def __init__(self, data_dir, batch_size, num_workers):
        super().__init__(d.CIFAR10, data_dir, batch_size, num_workers)

        self.name = "CIFAR"
        self.n_classes = 10

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.49139968, 0.48215841, 0.44653091),
                    (0.24703223, 0.24348513, 0.26158784),
                ),
            ]
        )
