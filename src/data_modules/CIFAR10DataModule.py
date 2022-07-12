import torchvision.datasets as d
from torchvision import transforms

from src.data_modules.BaseDataModule import BaseDataModule


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

    def prepare_data(self):
        super().prepare_data()
        d.DTD(self.data_dir, split='test', download=True)


    def setup(self, val_split=0.2, shuffle=True):
        super().setup(val_split, shuffle)

        # Set DTD (Describable Textures Dataset) as OOD dataset
        ood_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                # Found using self._compute_mean_and_std()
                transforms.Normalize(
                    (0.5287, 0.4742, 0.4236), 
                    (0.2588, 0.2499, 0.2553)
                ),
                transforms.Resize((32, 32)),
            ]
        )

        self.dataset_ood = d.DTD(
            self.data_dir, split='test', transform=ood_transforms
        )