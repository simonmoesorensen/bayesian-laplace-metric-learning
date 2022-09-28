from torchvision import transforms
import torchvision.datasets as d

import os, sys
from src.baselines.Laplace_online.utils import sample_nn

sys.path.append("../")
from src.baselines.models import CIFAR10ConvNet, FashionMNISTConvNet,FashionMNISTLinearNet
from src.baselines.PFE.models import FashionMNIST_PFE
from src.utils import filter_state_dict
import torch
from src.lightning_modules.BackboneLightningModule import BackboneLightningModule
from src.lightning_modules.PFELightningModule import PFELightningModule
from src.lightning_modules.PostHocLaplaceLightningModule import PostHocLaplaceLightningModule
from src.data_modules import (
    CIFAR10DataModule,
    FashionMNISTDataModule,
    MNISTDataModule,
)
# import optimizer
from torch.optim import Adam, SGD
from dotmap import DotMap
import matplotlib.pyplot as plt
import copy
from torch.nn.utils import parameters_to_vector, vector_to_parameters
import cv2
import torch.nn.functional as F


def softclamp(x, min=0, max=1):
    return (torch.tanh(x) + 1)/2

idx = 3
optimizer = Adam
pfe_model_path = "outputs/PFE/checkpoints/FashionMNIST/latentdim_3_seed_43_linear/"
pfe_model_name = "Final_Model_Epoch_200_Time_2022-09-27T121813_checkpoint.pth"
posthoc_model_path = "outputs/Laplace_posthoc/checkpoints/FashionMNIST/latent_dim_3_seed_43_linear/"
posthoc_model_name = "Final_Model_Epoch_1_Time_2022-09-27T115356_checkpoint.pth"
online_model_path = "outputs/Laplace_online/checkpoints/FashionMNIST/latentdim_3_seed_43_linear/"
online_model_name = "Final_Model_Epoch_200_Time_2022-09-27T104351_checkpoint.pth"

save_path = "rotate/"
os.makedirs(save_path, exist_ok=True)

args_all = {"latent_dim": 3,
        "data_dir": "/work3/frwa/datasets/",
        "dataset": "FashionMNIST",
        "batch_size": 32,
        "num_workers": 8,
        "gpu_id": [0],
        "random_seed": 43,
        "log_dir": "",
        "vis_dir": "",
        "linear": True}

args = DotMap(args_all)
data_module = FashionMNISTDataModule
data_module = data_module(
    args.data_dir,
    args.batch_size,
    args.num_workers,
    npos=1,
    nneg=0,
)


test_set = data_module.test_dataloader().dataset
original_image, label = test_set.__getitem__(idx)#data[idx:idx+1].float().unsqueeze(0) / 255.0
original_image = original_image.unsqueeze(0)

#### PFE ####    
pfe_model = FashionMNIST_PFE(embedding_size=args.latent_dim, linear=args.linear, seed=args.random_seed)
pfe_model.load_state_dict(torch.load(pfe_model_path + pfe_model_name))
pfe_model.eval()
pfe_model = pfe_model.cuda()

#### Post-hoc Laplace ####
model_name = "Posthoc"
if args.linear:
    posthoc_model = FashionMNISTLinearNet(latent_dim=args.latent_dim)
else:
    posthoc_model = FashionMNISTConvNet(latent_dim=args.latent_dim)

posthoc_trainer = PostHocLaplaceLightningModule(
    accelerator="gpu", devices=len(args.gpu_id), strategy="dp"
)

posthoc_trainer.add_data_module(data_module)
posthoc_trainer.n_test_samples = 100
posthoc_trainer.scale = torch.tensor(torch.load(posthoc_model_path + "scale.pth")).to("cuda:0")
posthoc_trainer.prior_prec = torch.load(posthoc_model_path + "prior_prec.pth").to("cuda:0")
posthoc_trainer.hessian = torch.load(posthoc_model_path + "hessian.pth").to("cuda:0")
posthoc_model.load_state_dict(torch.load(posthoc_model_path + posthoc_model_name))
posthoc_trainer.model = posthoc_model.to("cuda:0")

#### Online Laplace ####
model_name = "Online"
if args.linear:
    online_model = FashionMNISTLinearNet(latent_dim=args.latent_dim)
else:
    online_model = FashionMNISTConvNet(latent_dim=args.latent_dim)

online_trainer = PostHocLaplaceLightningModule(
    accelerator="gpu", devices=len(args.gpu_id), strategy="dp"
)

online_trainer.add_data_module(data_module)
online_trainer.n_test_samples = 100
online_trainer.scale = 1.0
online_trainer.prior_prec = 1.0
online_trainer.hessian = torch.load(online_model_path + "hessian.pth").to("cuda:0")
online_model.load_state_dict(torch.load(online_model_path + online_model_name))
online_trainer.model = online_model.to("cuda:0")

####
print("==> Who are better? :O ")


import numpy as np

results = []
for rot in range(0, 360, 5):
    theta = torch.tensor([[np.cos(np.deg2rad(rot)), -np.sin(np.deg2rad(rot)), 0], [np.sin(np.deg2rad(rot)), np.cos(np.deg2rad(rot)), 0]]).unsqueeze(0).float()
    grid = F.affine_grid(theta, original_image.size())
    rotated_image = F.grid_sample(original_image, grid)
    
    plt.figure()
    plt.imshow(rotated_image.squeeze().cpu().numpy())
    plt.savefig(save_path + f"rotated_image_{rot}.png")
    plt.close(); plt.cla(); plt.clf()
    
    _, online_z_sigma, _ = online_trainer.forward_samples(rotated_image.to("cuda:0"), 100)
    _, posthoc_z_sigma, _ = posthoc_trainer.forward_samples(rotated_image.to("cuda:0"), 100)
    _, pfe_z_sigma = pfe_model(rotated_image.to("cuda:0"))
    
    print("rotated", rot, online_z_sigma.sum().item(), posthoc_z_sigma.sum().item(), pfe_z_sigma.sum().item())
    results.append([rot, online_z_sigma.sum().item(), posthoc_z_sigma.sum().item(), pfe_z_sigma.sum().item()])
    
results = np.array(results)
plt.plot(results[:,0], results[:,1], "-o", label="online")
plt.plot(results[:,0], results[:,2], "-o", label="posthoc")
plt.plot(results[:,0], results[:,3], "-o", label="pfe")
plt.legend()
plt.savefig(save_path + "results.png")