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
type = "max_sigma"
optimizer = Adam
lr = 1e-3
steps = 1000
init_noise = 0 #.3 * torch.randn(1, 2,3).cuda()
pfe_model_path = "outputs/PFE/checkpoints/FashionMNIST/latentdim_3_seed_43_linear/"
pfe_model_name = "Final_Model_Epoch_200_Time_2022-09-27T121813_checkpoint.pth"
posthoc_model_path = "outputs/Laplace_posthoc/checkpoints/FashionMNIST/latent_dim_3_seed_43_linear/"
posthoc_model_name = "Final_Model_Epoch_1_Time_2022-09-27T115356_checkpoint.pth"
online_model_path = "outputs/Laplace_online/checkpoints/FashionMNIST/latentdim_3_seed_43_linear/"
online_model_name = "Final_Model_Epoch_200_Time_2022-09-27T104351_checkpoint.pth"

save_path = "optimize/{optimizer.__name__}_{lr}/{type}/".format(optimizer=optimizer, lr=lr,type=type)
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

if idx == "noise":
    original_image = torch.randn(1,1,28,28)
    original_image = softclamp(original_image, 0, 1)
else:
    test_set = data_module.test_dataloader().dataset
    original_image, label = test_set.__getitem__(idx)#data[idx:idx+1].float().unsqueeze(0) / 255.0
    original_image = original_image.unsqueeze(0)

#### PFE ####    
print("==> PFE")
print("=> ORIGINAL IMAGE ", original_image.min(), original_image.max(), original_image.mean(), original_image.median())

pfe_model = FashionMNIST_PFE(embedding_size=args.latent_dim, linear=args.linear, seed=args.random_seed)
pfe_model.load_state_dict(torch.load(pfe_model_path + pfe_model_name))
pfe_model.eval()
pfe_model = pfe_model.cuda()

for param in pfe_model.parameters():
    param.requires_grad = False

pfe_image = copy.deepcopy(original_image)
pfe_image = pfe_image.cuda()

if type == "affine":
    theta = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]).unsqueeze(0).cuda()
    theta += init_noise
    theta = torch.nn.Parameter(theta)
    optim = optimizer([theta], lr=lr)
else:
    pfe_image.requires_grad = True
    pfe_image = torch.nn.Parameter(pfe_image)
    optim = optimizer([pfe_image], lr=lr)

losses = []
for iter in range(steps):
    optim.zero_grad()
    
    if type == "affine":
        grid = F.affine_grid(theta, pfe_image.size())
        image_01 = F.grid_sample(pfe_image, grid)
    else:
        image_01 = softclamp(pfe_image, 0, 1)
    mu, sigma = pfe_model(image_01)
    
    loss = -sigma.sum()
    loss.backward()
    
    optim.step()

    if iter % 100 == 0:
        print(f"iter {iter} loss {loss} sigma {sigma.sum().item()}")
    
    losses.append(loss.cpu().item())
    
plt.plot(losses)
plt.savefig(save_path + f"{idx}_pfe_loss.png")
plt.close(); plt.clf(); plt.cla()

if type == "affine":
    grid = F.affine_grid(theta, pfe_image.size())
    optimized_pfe_image = F.grid_sample(pfe_image, grid)
else:
    optimized_pfe_image = softclamp(pfe_image, 0, 1).detach()
print("=> OPTIMIZED IMAGE ", optimized_pfe_image.min(), optimized_pfe_image.max(), optimized_pfe_image.mean(), optimized_pfe_image.median())
cv2.imwrite(save_path + f"{idx}_pfe_optimized.png", optimized_pfe_image.detach().cpu().numpy().squeeze() * 255)
cv2.imwrite(save_path + f"{idx}_pfe_original.png", original_image.cpu().numpy().squeeze() * 255)
"""
plt.imshow(original_image.cpu().numpy().squeeze(), cmap="gray")
plt.savefig(save_path + f"{idx}_pfe_original.png")
plt.axis("off")
plt.grid(None) 
plt.close(); plt.clf(); plt.cla()
    
optimized_pfe_image = torch.sigmoid(pfe_image).detach()
plt.imshow(optimized_pfe_image.cpu().numpy().squeeze(), cmap="gray")
plt.savefig(save_path + f"{idx}_pfe_optimized.png")
plt.axis("off")
plt.grid(None) 
plt.close(); plt.clf(); plt.cla()
"""
#### Post-hoc Laplace ####
print("==> Post-hoc Laplace")
print("=> ORIGINAL IMAGE ", original_image.min(), original_image.max(), original_image.mean(), original_image.median())

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

posthoc_image = copy.deepcopy(original_image)
posthoc_image = posthoc_image.cuda()

if type == "affine":
    theta = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], requires_grad=True).unsqueeze(0).cuda()
    theta += init_noise
    theta = torch.nn.Parameter(theta)
    optim = optimizer([theta], lr=lr)
else:
    posthoc_image.requires_grad = True
    posthoc_image = torch.nn.Parameter(posthoc_image)
    optim = optimizer([posthoc_image], lr=lr)

for param in posthoc_model.parameters():
    param.requires_grad = False

mu_q = parameters_to_vector(posthoc_model.linear.parameters())
mu_q.requires_grad = False
sigma_q = 1 / (posthoc_trainer.hessian * posthoc_trainer.scale + posthoc_trainer.prior_prec).sqrt()
sigma_q.requires_grad = False

losses = []
for iter in range(steps):
    optim.zero_grad()
    
    if type == "affine":
        grid = F.affine_grid(theta, posthoc_image.size())
        image_01 = F.grid_sample(posthoc_image, grid)
    else:
        image_01 = softclamp(posthoc_image, 0, 1)
        
    z_mu, sigma, z = posthoc_trainer.forward_samples(image_01, 100)
        
    loss = -sigma.sum()
    loss.backward()
    optim.step()
    
    if iter % 100 == 0:
        print(f"iter {iter} loss {loss} sigma {sigma.sum().item()}")
    
    losses.append(loss.item())
    
plt.plot(losses)
plt.savefig(save_path + f"{idx}_laml_loss.png")
plt.close(); plt.clf(); plt.cla()

if type == "affine":
    grid = F.affine_grid(theta, posthoc_image.size())
    optimized_posthoc_image = F.grid_sample(posthoc_image, grid)
else:
    optimized_posthoc_image = softclamp(posthoc_image, 0, 1).detach()

print("=> OPTIMIZED IMAGE ", optimized_posthoc_image.min(), optimized_posthoc_image.max(), optimized_posthoc_image.mean(), optimized_posthoc_image.median())
cv2.imwrite(save_path + f"{idx}_laml_optimized.png", optimized_posthoc_image.detach().cpu().numpy().squeeze() * 255)
cv2.imwrite(save_path + f"{idx}_laml_original.png", original_image.cpu().numpy().squeeze() * 255)

"""

plt.imshow(optimized_posthoc_image.cpu().numpy().squeeze(), cmap="gray")
plt.savefig(save_path + f"{idx}_laml_optimized.png")
plt.axis("off")
plt.grid(None) 
plt.close(); plt.clf(); plt.cla()

plt.imshow(original_image.cpu().numpy().squeeze(), cmap="gray")
plt.savefig(save_path + f"{idx}_laml_original.png")
plt.axis("off")
plt.grid(None) 
plt.close(); plt.clf(); plt.cla()
"""

### Online laplace ###
print("==> Online Laplace")
print("=> ORIGINAL IMAGE ", original_image.min(), original_image.max(), original_image.mean(), original_image.median())

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

online_image = copy.deepcopy(original_image)
online_image = online_image.cuda()

if type == "affine":
    theta = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], requires_grad=True).unsqueeze(0).cuda()
    theta += init_noise
    theta = torch.nn.Parameter(theta)
    optim = optimizer([theta], lr=lr)
else:
    online_image.requires_grad = True
    online_image = torch.nn.Parameter(online_image)
    optim = optimizer([online_image], lr=lr)

for param in online_model.parameters():
    param.requires_grad = False

mu_q = parameters_to_vector(online_model.linear.parameters())
mu_q.requires_grad = False
sigma_q = 1 / (online_trainer.hessian * online_trainer.scale + online_trainer.prior_prec).sqrt()
sigma_q.requires_grad = False

losses = []
for iter in range(steps):
    optim.zero_grad()
    
    if type == "affine":
        grid = F.affine_grid(theta, online_image.size())
        image_01 = F.grid_sample(online_image, grid)
    else:
        image_01 = softclamp(online_image, 0, 1)
    
    z_mu, sigma, z = online_trainer.forward_samples(image_01, 100)
    
    loss = -sigma.sum()
    loss.backward()
    optim.step()

    if iter % 100 == 0:
        print(f"iter {iter} loss {loss} sigma {sigma.sum().item()}")
    
    losses.append(loss.item())
    
plt.plot(losses)
plt.savefig(save_path + f"{idx}_olaml_loss.png")
plt.close(); plt.clf(); plt.cla()


if type == "affine":
    grid = F.affine_grid(theta, online_image.size())
    optimized_online_image = F.grid_sample(online_image, grid)
else:
    optimized_online_image = softclamp(online_image, 0, 1).detach()
    
print("=> OPTIMIZED IMAGE ", optimized_online_image.min(), optimized_online_image.max(), optimized_online_image.mean(), optimized_online_image.median())
cv2.imwrite(save_path + f"{idx}_olaml_optimized.png", optimized_online_image.detach().cpu().numpy().squeeze() * 255)
cv2.imwrite(save_path + f"{idx}_olaml_original.png", original_image.cpu().numpy().squeeze() * 255)

"""
plt.imshow(optimized_online_image.cpu().numpy().squeeze(), cmap="gray")
plt.axis("off")
plt.grid(None) 
plt.savefig(save_path + f"{idx}_olaml_optimized.png")

plt.imshow(original_image.cpu().numpy().squeeze(), cmap="gray")
plt.axis("off")
plt.grid(None) 
plt.savefig(save_path + f"{idx}_olaml_original.png")
"""

####
print("==> Who are better? :O ")

print("==> PFE")
z_mu, z_sigma = pfe_model(original_image.to("cuda:0"))
print("original", z_sigma.sum().item())
z_mu, z_sigma = pfe_model(optimized_online_image.to("cuda:0"))
print("online", z_sigma.sum().item())
z_mu, z_sigma = pfe_model(optimized_posthoc_image.to("cuda:0"))
print("posthoc", z_sigma.sum().item())

print("==> POSTHOC")
z_mu, z_sigma, z_samples = posthoc_trainer.forward_samples(original_image.to("cuda:0"), 100)
print("original", z_sigma.sum().item())
z_mu, z_sigma, z_samples = posthoc_trainer.forward_samples(optimized_online_image.to("cuda:0"), 100)
print("online optimized", z_sigma.sum().item())
z_mu, z_sigma, z_samples = posthoc_trainer.forward_samples(optimized_pfe_image.to("cuda:0"), 100)
print("pfe optimized", z_sigma.sum().item())

print("==> ONLINE")
z_mu, z_sigma, z_samples = online_trainer.forward_samples(original_image.to("cuda:0"), 100)
print("original", z_sigma.sum().item())
z_mu, z_sigma, z_samples = online_trainer.forward_samples(optimized_posthoc_image.to("cuda:0"), 100)
print("posthoc optimized", z_sigma.sum().item())
z_mu, z_sigma, z_samples = online_trainer.forward_samples(optimized_pfe_image.to("cuda:0"), 100)
print("pfe optimized", z_sigma.sum().item())