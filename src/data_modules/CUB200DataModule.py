import torchvision.datasets as d
from src.data_modules.BaseDataModule import BaseDataModule
from torchvision import transforms
from torch.utils.data import Dataset
import numpy as np
import time
import torch
from PIL import Image

class Cub200_train(Dataset):
    def __init__(self, data_dir, npos=1, nneg=5):
        
        dataset_path = data_dir / 'CUB_200_2011'
        
        image_names = np.loadtxt(dataset_path / 'images.txt', dtype=str)
        image_class_labels = np.loadtxt(dataset_path / 'image_class_labels.txt', dtype=int)
        self.image_path = dataset_path / "images"
        self.labels = image_class_labels[:,1]
        self.images = image_names[:,1]
        
        idx = self.labels <= 100
        self.labels = self.labels[idx]
        self.images = self.images[idx]
        
        self.npos = npos
        self.nneg = nneg
        self.classes = np.unique(self.labels)
        
        print("=> this is a bit slow, but only done once")
        t = time.time()
        
        self.idx = {}
        for c in self.classes:
            self.idx[f"{c}"] = {"pos": np.where(self.labels == c)[0], "neg": np.where(self.labels != c)[0]}
        
        self.pos_idx = []
        self.neg_idx = []
        for i in range(len(self.labels)):
            key = f"{self.labels[i]}"
            
            pos_idx = self.idx[key]["pos"] 
            pos_idx = pos_idx[pos_idx != i] # remove self
            
            neg_idx = self.idx[key]["neg"]    
            
            self.pos_idx.append(pos_idx)
            self.neg_idx.append(neg_idx)
        print("=> done in {:.2f}s".format(time.time() - t))
        
        self.transform = transforms.Compose([
            transforms.Resize(156),
            transforms.RandomCrop(128),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        
    def __len__(self):
        return len(self.labels)

    def load_image(self, path):
        im = Image.open(self.image_path / path).convert('RGB')
        return self.transform(im)

    def __getitem__(self, idx):
                
        a = self.labels[idx]
        images = [self.load_image(self.images[idx])]
        labels = [0]
        classes = [a]
        
        if self.npos > 0:
            p = np.random.choice(self.pos_idx[idx], self.npos, replace=False)
            for lp in self.labels[p]: assert a == lp
            images += [self.load_image(self.images[pi]) for pi in p]
            labels += [1] * self.npos
            classes += [lp for lp in self.labels[p]]
                
        if self.nneg > 0:
            n = np.random.choice(self.neg_idx[idx], self.nneg, replace=False)
            for ln in self.labels[n]: assert a != ln
            images += [self.load_image(self.images[ni]) for ni in n]
            labels += [-1] * self.nneg
            classes += [ln for ln in self.labels[n]]
                 
        images = torch.stack(images, dim=0).float()
        labels = torch.from_numpy(np.stack(labels))
        classes = torch.from_numpy(np.stack(classes))
                
        assert len(images) == len(labels)
        
        return images, labels, classes
    
class Cub200_test(Dataset):
    def __init__(self, data_dir, npos=1, nneg=5):
  
        dataset_path = data_dir / 'CUB_200_2011'
        self.image_path = dataset_path / "images"
        image_names = np.loadtxt(dataset_path / 'images.txt', dtype=str)
        image_class_labels = np.loadtxt(dataset_path / 'image_class_labels.txt', dtype=int)
        
        self.labels = image_class_labels[:,1]
        self.images = image_names[:,1]
        
        idx = self.labels > 100
        self.labels = self.labels[idx]
        self.images = self.images[idx]
        
        self.transform = transforms.Compose([
            transforms.Resize(156),
            transforms.CenterCrop(128),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        
        image = self.transform(Image.open(self.image_path / self.images[idx]).convert('RGB'))
        label = self.labels[idx]
                
        return image, label


class CUB200DataModule(BaseDataModule):
    def __init__(
        self,
        data_dir,
        batch_size,
        num_workers,
        npos=1,
        nneg=5,
    ):
        super().__init__(
            data_dir,
            batch_size,
            num_workers,
        )

        self.name = "CUB200"
        self.n_classes = 100
        
        self.dataset_train = Cub200_train(data_dir, nneg=nneg, npos=npos)
        self.dataset_val = Cub200_test(data_dir)
        self.dataset_test = Cub200_test(data_dir)
        
        self.dataset_ood = None
