from pathlib import Path
from torchvision import transforms
import torchvision.datasets as d
from torch.utils.data import Subset, random_split, DataLoader, sampler
import zipfile
import numpy as np
import torch

from tqdm import tqdm

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split, Subset
import numpy as np
from tqdm import tqdm
import torch

import requests
from tqdm import tqdm


def download_file_from_google_drive(id, destination):
    """
    https://stackoverflow.com/questions/25010369/wget-curl-large-file-from-google-drive/39225039#39225039
    """

    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith("download_warning"):
                return value

        return None

    def save_response_content(response, destination):
        CHUNK_SIZE = 32768

        total = int(response.headers.get('content-length', 0))

        with open(destination, "wb") as f, tqdm(total=total, unit='iB', unit_scale=True, unit_divisor=1024) as pbar:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk:  # filter out keep-alive new chunks
                    size = f.write(chunk)
                    pbar.update(size)

    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params={"id": id, 'confirm': 't'}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {"id": id, "confirm": token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)

    
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


class CasiaDataModule(BaseDataModule):
    def __init__(
        self,
        data_dir,
        batch_size,
        num_workers,
        sampler=None,
        shuffle=False,
        pin_memory=True,
    ):
        data_dir = Path(data_dir)
        super().__init__(
            d.ImageFolder,
            data_dir,
            batch_size,
            num_workers,
            sampler,
            shuffle,
            pin_memory,
        )

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
                    [0.2949, 0.2649, 0.2588],
                ),
                transforms.RandomCrop((128, 128), padding=16),
                transforms.RandomHorizontalFlip(0.5),
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

        print(f"Training set size: {n_train}")
        print(f"Validation set size: {n_val}")
        print(f"Test set size: {n_test}")

        # Ensure that the splits cover the whole dataset
        n_train += size - n_train - n_val - n_test

        if shuffle:
            # Overlapping classes allowed
            self.dataset_train, self.dataset_val, self.dataset_test = random_split(
                dataset_full,
                [n_train, n_val, n_test],
                generator=torch.Generator().manual_seed(42),
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
                # transforms.RandomCrop((128, 128), pad_if_needed=True),
                # transforms.RandomHorizontalFlip(0.5),
            ]
        )

        self.dataset_ood = d.CIFAR10(
            self.data_dir, train=False, transform=ood_transforms
        )
        
        # OOD noise
        ood_noise_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    # Found from `self._compute_mean_and_std()`
                    [0.4668, 0.3803, 0.3344],
                    [0.2949, 0.2649, 0.2588],
                ),
                # transforms.RandomCrop((128, 128), padding=16),
                # transforms.RandomHorizontalFlip(0.5),
                transforms.GaussianBlur(kernel_size=31, sigma=5),
            ]
        )

        ood_full = d.ImageFolder(self.img_path, transform=ood_noise_transforms)
        ood_size = 10000
        print(f"OOD set size: {ood_size}")

        self.dataset_ood, _ = random_split(
            ood_full,
            [ood_size, size - ood_size]
        )

        if self.sampler == "WeightedRandomSampler":
            weights_file = self.data_dir / "casia_weights.tensor"

            if weights_file.exists():
                print(f"Found weights file {weights_file}")
                weights = torch.load(weights_file)
            else:
                print("Computing class weights")
                weights = self.make_weights_for_balanced_classes(
                    self.dataset_train, self.n_classes
                )
                print(f"Saving class weights file {weights_file}")
                torch.save(weights, weights_file)

            self.sampler = self.get_sampler(weights)

    def ood_dataloader(self):
        return DataLoader(
            self.dataset_ood,
            num_workers=self.num_workers,
            batch_size=128,
            pin_memory=self.pin_memory,
        )

    def make_weights_for_balanced_classes(self, images, nclasses):
        """
        Adapted from https://gist.github.com/srikarplus/15d7263ae2c82e82fe194fc94321f34e
        """

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        batch_size = 1024
        num_workers = self.num_workers

        count = torch.zeros(nclasses).to(device)
        loader = DataLoader(images, batch_size=batch_size, num_workers=num_workers)

        for _, label in tqdm(loader, desc="Counting classes"):
            label = label.to(device=device)
            idx, counts = label.unique(return_counts=True)
            count[idx] += counts

        N = count.sum()
        weight_per_class = N / count

        weight = torch.zeros(len(images)).to(device)

        for i, (img, label) in tqdm(
            enumerate(loader), desc="Apply weights", total=len(loader)
        ):
            idx = torch.arange(0, img.shape[0]) + (i * batch_size)
            idx = idx.to(dtype=torch.long, device=device)
            weight[idx] = weight_per_class[label]

        return weight

    def get_sampler(self, weights):
        return sampler.WeightedRandomSampler(weights, len(weights))
