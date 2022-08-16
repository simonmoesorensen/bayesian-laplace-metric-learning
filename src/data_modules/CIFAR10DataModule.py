import torchvision.datasets as d
from torchvision import transforms

from src.data_modules.BaseDataModule import BaseDataModule


class CIFAR10DataModule(BaseDataModule):
    def __init__(
        self,
        data_dir,
        batch_size,
        num_workers,
        sampler=None,
        shuffle=False,
        pin_memory=True,
    ):
        super().__init__(
            d.CIFAR10, data_dir, batch_size, num_workers, sampler, shuffle, pin_memory
        )

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
        d.SVHN(self.data_dir, split="test", download=True)

    def setup(self, val_split=0.2, shuffle=True):
        super().setup(val_split, shuffle)

        # Set SVHN (Street View House Numbers) as OOD dataset
        ood_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                # Found using self._compute_mean_and_std()
                transforms.Normalize(
                    (0.4376821, 0.4437697, 0.47280442),
                    (0.19803012, 0.20101562, 0.19703614),
                ),
            ]
        )

        self.dataset_ood = d.SVHN(self.data_dir, split="test", transform=ood_transforms)

        # # OOD noise
        # ood_noise_transforms = transforms.Compose(
        #     [
        #         transforms.ToTensor(),
        #         transforms.Normalize(
        #             (0.49139968, 0.48215841, 0.44653091),
        #             (0.24703223, 0.24348513, 0.26158784),
        #         ),
        #         transforms.RandomCrop((32, 32), padding=4),
        #         transforms.RandomHorizontalFlip(0.5),
        #         transforms.GaussianBlur(
        #             kernel_size=7,
        #             sigma=3,
        #         ),
        #     ]
        # )

        # self.dataset_ood = d.CIFAR10(
        #     self.data_dir, train=False, transform=ood_noise_transforms
        # )
