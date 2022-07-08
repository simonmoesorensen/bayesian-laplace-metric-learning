from pathlib import Path
from torchvision import transforms
import torchvision.datasets as d
from torch.utils.data import Subset, random_split
import zipfile

from tqdm import tqdm
from gdrive import download_file_from_google_drive

from BaseDataModule import BaseDataModule

class CasiaDataModule(BaseDataModule):
    def __init__(self, data_dir, batch_size, num_workers):
        super().__init__(d.ImageFolder, data_dir, batch_size, num_workers)

        self.name = "Cassia"
        self.n_classes = 10575

        self.img_path = self.data_dir / "CASIA-WebFace"

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                transforms.Resize((64, 64)),
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
                        zf.extract(member, self.img_path)
                    except zipfile.error as e:
                        print(e)
                        pass

    def setup(self, stage=None, val_split=0.2, test_split=0.2, shuffle=False):
        assert self.transform, "transform must be set before setup()"

        dataset_full = self.cls(self.img_path, transform=self.transform)
        size = len(dataset_full)

        def get_split_size(frac):
            return int(size // (1 / frac))

        n_train = get_split_size(1 - val_split - test_split)
        n_val = get_split_size(1 - val_split)
        n_test = get_split_size(1 - test_split)

        if shuffle:
            self.dataset_train, self.dataset_val, self.dataset_test = random_split(
                dataset_full, [n_train, n_val, n_test]
            )
        else:
            self.dataset_train = Subset(dataset_full, range(0, n_train))
            self.dataset_val = Subset(dataset_full, range(n_train, n_val))
            self.dataset_test = Subset(dataset_full, range(n_val, n_test))


if __name__ == "__main__":
    data_dir = Path("/work3/s174420/datasets")

    data = CasiaDataModule(data_dir, batch_size=64, num_workers=12)
    data.prepare_data()

    data.setup()

    print(next(iter(data.train_dataloader())))
