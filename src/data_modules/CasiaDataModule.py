from torchvision import transforms
import torchvision.datasets as d
from torch.utils.data import Subset, random_split, DataLoader
import zipfile
import numpy as np

from tqdm import tqdm

from src.data_modules.gdrive import download_file_from_google_drive
from src.data_modules.BaseDataModule import BaseDataModule


class CasiaDataModule(BaseDataModule):
    def __init__(self, data_dir, batch_size, num_workers, sampler=None, shuffle=False, pin_memory=True):
        super().__init__(d.ImageFolder, data_dir, batch_size, num_workers, sampler, shuffle, pin_memory)

        self.name = "Casia"
        self.n_classes = 10575

        # Used to reference the extracted data
        self.img_path = self.data_dir / "CASIA-WebFace"

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    # Found from `self._compute_mean_and_std()`
                    [0.4668, 0.3803, 0.3344], 
                    [0.2949, 0.2649, 0.2588]
                ),
                transforms.Resize((128, 128)),
            ]
        )

    def prepare_data(self):
        file_id = "1Of_EVz-yHV7QVWQGihYfvtny9Ne8qXVz"
        file_path = self.data_dir / "casia.zip"

        if zipfile.is_zipfile(file_path):
            print("Files already downloaded")
        else:
            print(f"Downloading casia file to {file_path}")
            download_file_from_google_drive(file_id, file_path)

        if self.img_path.exists():
            print(f"Files already extracted in {self.img_path}")
        else:
            print(f"Extracting {file_path} to {self.img_path}")
            with zipfile.ZipFile(file_path, "r") as zf:
                for member in tqdm(zf.infolist(), desc="Extracting "):
                    try:
                        # Creates the 'CASIA-WebFace' directory in self.data_dir
                        zf.extract(member, self.data_dir)
                    except zipfile.error as e:
                        print(e)
                        pass
        
        # OOD dataset
        d.CIFAR10(self.data_dir, train=False, download=True)

    def setup(self, val_split=0.2, test_split=0.2, shuffle=True):
        assert self.transform, "transform must be set before setup()"

        dataset_full = self.cls(self.img_path, transform=self.transform)
        size = len(dataset_full)

        def get_split_size(frac):
            return np.round(size // (1 / frac)).astype(int)

        n_train = get_split_size(1 - val_split - test_split)
        n_val = get_split_size(val_split)
        n_test = get_split_size(test_split)

        # Ensure that the splits cover the whole dataset
        n_train += size - n_train - n_val - n_test

        if shuffle:
            # Overlapping classes allowed
            self.dataset_train, self.dataset_val, self.dataset_test = random_split(
                dataset_full, [n_train, n_val, n_test]
            )
        else:
            # Overlapping classes not allowed (zero-shot learning)
            self.dataset_train = Subset(dataset_full, range(0, n_train))
            self.dataset_val = Subset(dataset_full, range(n_train, n_train + n_val))
            self.dataset_test = Subset(
                dataset_full, range(n_train + n_val, n_train + n_val + n_test)
            )

        # Set OOD dataset
        ood_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    # Found from `self._compute_mean_and_std()`
                    (0.49139968, 0.48215841, 0.44653091),
                    (0.24703223, 0.24348513, 0.26158784),
                ),
                transforms.Resize((32, 32)),
            ]
        )

        self.dataset_ood = d.CIFAR10(self.data_dir, train=False, transform=ood_transforms)


    def ood_dataloader(self):
        return DataLoader(
            self.dataset_ood,
            num_workers=self.num_workers,
            batch_size=128,
            pin_memory=self.pin_memory,
            sampler=self.sampler,
            shuffle=False if self.sampler else self.shuffle,
        )