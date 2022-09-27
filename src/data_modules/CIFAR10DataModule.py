import torchvision.datasets as d
from src.data_modules.BaseDataModule import BaseDataModule
from torchvision import transforms
from torch.utils.data import Dataset
import numpy as np
import time
import torch

class Cifar10(Dataset):
    def __init__(self, images, labels, npos=1, nneg=5):
        self.images = images
        self.labels = labels
        self.transform = None
        
        self.npos = npos
        self.nneg = nneg
        self.classes = np.unique(labels)
        
        print("=> this is a bit slow, but only done once")
        t = time.time()
        
        self.idx = {}
        for c in self.classes:
            self.idx[f"{c}"] = {"pos": np.where(self.labels == c)[0], "neg": np.where(self.labels != c)[0]}
        
        self.pos_idx = []
        self.neg_idx = []
        for i in range(len(self.labels)):
            key = f"{self.labels[i].data}"
            
            pos_idx = self.idx[key]["pos"] 
            pos_idx = pos_idx[pos_idx != i] # remove self
            
            neg_idx = self.idx[key]["neg"]    
            
            self.pos_idx.append(pos_idx)
            self.neg_idx.append(neg_idx)
        print("=> done in {:.2f}s".format(time.time() - t))
        
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        
        a = self.labels[idx]
        images = [self.images[idx]]
        labels = [0]
        classes = [a]
        
        if self.npos > 0:
            p = np.random.choice(self.pos_idx[idx], self.npos, replace=False)
            for lp in self.labels[p]: assert a == lp
            images += [self.images[pi] for pi in p]
            labels += [1] * self.npos
            classes += [lp for lp in self.labels[p]]
                
        if self.nneg > 0:
            n = np.random.choice(self.neg_idx[idx], self.nneg, replace=False)
            for ln in self.labels[n]: assert a != ln
            images += [self.images[ni] for ni in n]
            labels += [-1] * self.nneg
            classes += [ln for ln in self.labels[n]]
                 
        if self.transform:
            images = [self.transform(im) for im in images]
        
        images = torch.stack(images, dim=0).permute(0, 3, 1, 2).float()
        labels = torch.from_numpy(np.stack(labels))
        classes = torch.from_numpy(np.stack(classes))
        
        assert len(images) == len(labels)
        
        return images, labels, classes

class CIFAR10DataModule(BaseDataModule):
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

        self.name = "CIFAR"
        self.n_classes = 10
        
        dataset_train = d.CIFAR10(data_dir, train=True, download=True, transform=transforms.ToTensor())
        self.dataset_train = Cifar10(torch.from_numpy(dataset_train.data / 255.0), torch.from_numpy(np.array(dataset_train.targets)), npos, nneg)
        
        self.dataset_test = d.CIFAR10(data_dir, train=False, download=True, transform=transforms.ToTensor())
        self.dataset_val = d.CIFAR10(data_dir, train=False, download=True, transform=transforms.ToTensor())
        self.dataset_ood = d.SVHN(data_dir, split="test", download=True, transform=transforms.ToTensor())
