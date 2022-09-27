import torch.nn as nn
import torch

from src.utils import L2Norm
import torchvision

class SampleNet(nn.Module):
    def pass_through(self, x):
        """ " Normal forward pass"""
        raise NotImplementedError()

    def sample(self, X, samples):
        zs = []

        for _ in range(samples):
            zs.append(self.pass_through(X))

        zs = torch.stack(zs, dim=-1)

        return zs.mean(dim=-1), zs.std(dim=-1), zs

    def forward(self, x, samples=100):
        if samples:
            return self.sample(x, samples)
        else:
            return self.pass_through(x)


class CIFAR10ConvNet(nn.Module):
    def __init__(self, latent_dim=128, p=0):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(16, 32, 3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(32, 64, 3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(64, 64, 3,padding=1),
            nn.Dropout2d(p),
            nn.Flatten(),
        )
        self.linear = nn.Sequential(
            nn.Linear(1024, 512),
            nn.Tanh(),
            nn.Linear(512, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, latent_dim),
            L2Norm(),
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.linear(x)
        return x
    
class CIFAR10LinearNet(nn.Module):
    def __init__(self, latent_dim=128, p=0):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Flatten(),
        )
        self.linear = nn.Sequential(
            nn.Linear(3 * 32 * 32, 512),
            #nn.Tanh(),
            #nn.Linear(1024, 512),
            nn.Tanh(),
            nn.Linear(512, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, latent_dim),
            L2Norm(),
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.linear(x)
        return x
    
    


class FashionMNISTConvNet(nn.Module):
    def __init__(self, latent_dim=32, p=0):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 8, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=0),
            nn.ReLU(),
            nn.Dropout2d(p),
            nn.Flatten(),
        )
        self.linear = nn.Sequential(
            nn.Linear(3 * 3 * 32, 128),
            nn.Tanh(),
            nn.Linear(128, latent_dim),
            L2Norm(),
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.linear(x)
        return x



class FashionMNISTLinearNet(nn.Module):
    def __init__(self, latent_dim=32, p=0):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Flatten(),
        )
        self.linear = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.Tanh(),
            nn.Linear(512, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, latent_dim),
            L2Norm(),
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.linear(x)
        return x

class CUB200ConvNet(nn.Module):
    def __init__(self, latent_dim=32, p=0):
        super().__init__()
        
        resnet = torchvision.models.resnet18(pretrained=True)
        self.conv = nn.Sequential(*(list(resnet.children())[:-1] + [nn.Flatten()]))
        self.linear = nn.Sequential(
            nn.Linear(512, 512),
            nn.Tanh(),
            nn.Linear(512, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, latent_dim),
            L2Norm(),
        )

    def forward(self, x):
        
        x = self.conv(x)
        x = self.linear(x)
        
        return x