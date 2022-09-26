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
from torch.optim import Adam
from dotmap import DotMap
import matplotlib.pyplot as plt
import copy
from torch.nn.utils import parameters_to_vector, vector_to_parameters


args_all = {"latent_dim": 3,
        "data_dir": "/work3/frwa/datasets/",
        "dataset": "FashionMNIST",
        "batch_size": 32,
        "num_workers": 8,
        "gpu_id": [0],
        "model": "PFE",
        "random_seed": 42,
        "log_dir": "",
        "vis_dir": ""}


if args_all["model"] == "PFE":
    args_all["model_path"] = "outputs/PFE/checkpoints/FashionMNIST/latent_dim_3_seed_42_conv/Model_Epoch_20_Time_2022-09-25T165518_checkpoint.pth"
    module = PFELightningModule

args = DotMap(args_all)

if args.model == "PFE":
    model = FashionMNIST_PFE(embedding_size=args.latent_dim)
else:
    model = FashionMNISTConvNet(latent_dim=args.latent_dim)
    
data_module = FashionMNISTDataModule

data_module = data_module(
    args.data_dir,
    args.batch_size,
    args.num_workers,
    npos=1,
    nneg=0,
)


model.load_state_dict(torch.load(args.model_path))
model.eval()

for param in model.parameters():
    param.requires_grad = False

# image = torch.randn(1,1,28,28, requires_grad=True)

idx = 1
test_set = data_module.test_dataloader().dataset
original_image = test_set.data[idx:idx+1].float().unsqueeze(0) / 255.0
image = copy.deepcopy(original_image)
image.requires_grad = True

path = "optimize/PFE/"
os.makedirs(path, exist_ok=True)

plt.imshow(torch.sigmoid(image).detach().numpy().squeeze(), cmap="gray")
plt.savefig(path + f"{idx}_original.png")

optimizer = Adam([image], lr=0.01)

for iter in range(100):
    optimizer.zero_grad()
    
    image_01 = torch.sigmoid(image)
    mu, sigma = model(image_01)
    
    loss = sigma.sum()
    loss.backward()
    optimizer.step()

    print(f"iter {iter} loss {loss} sigma {sigma.sum().item()}")
    

optimized_image = torch.sigmoid(image).detach()
plt.imshow(optimized_image.numpy().squeeze(), cmap="gray")
plt.savefig(path + f"{idx}_optimized.png")

##
# How does laplace handle these images

model_name = "Posthoc"
model = FashionMNISTConvNet(latent_dim=args.latent_dim)

trainer = PostHocLaplaceLightningModule(
    accelerator="gpu", devices=len(args.gpu_id), strategy="dp"
)

trainer.add_data_module(data_module)
trainer.n_test_samples = 100

if model_name in ("Posthoc", "Online"):
    if model_name == "Posthoc":
        path = "outputs/Laplace_posthoc/checkpoints/FashionMNIST/latent_dim_3_seed_42_opt_conv/"
        trainer.scale = torch.tensor(torch.load(path + "scale.pth")).to("cuda:0")
        trainer.prior_prec = torch.load(path + "prior_prec.pth").to("cuda:0")
        trainer.hessian = torch.load(path + "hessian.pth").to("cuda:0")
        model.load_state_dict(torch.load(path + "Final_Model_Epoch_1_Time_2022-09-25T163503_checkpoint.pth"))
        trainer.model = model.to("cuda:0")
        #trainer.scale = torch.tensor(1.0).to("cuda:0")
        #trainer.prior_prec = torch.tensor(1.0).to("cuda:0")
    else:
        raise NotImplementedError
    

z_mu, z_sigma, z_samples = trainer.forward_samples(original_image.to("cuda:0"), 100)
print("original", z_sigma.sum().item())
z_mu, z_sigma, z_samples = trainer.forward_samples(optimized_image.to("cuda:0"), 100)
print("optimized", z_sigma.sum().item())


####
# symmetric experiment
####

image = copy.deepcopy(original_image)
image = image.cuda()
image.requires_grad = True
optimizer = Adam([image], lr=0.01)

mu_q = parameters_to_vector(model.linear.parameters())
sigma_q = 1 / (trainer.hessian * trainer.scale + trainer.prior_prec).sqrt()
#sigma_q = 1 / (trainer.hessian * 1 + 1).sqrt()

for iter in range(100):
    optimizer.zero_grad()
    
    image_01 = torch.sigmoid(image)
    tmp = model.conv(image_01)
    samples = sample_nn(mu_q, sigma_q, 100)
    
    z = []
    for sample in samples:
        vector_to_parameters(sample, model.linear.parameters())
        z_i = model.linear(tmp)
        z.append(z_i)
        
    z = torch.cat(z, dim=0)
    mu = z.mean(dim=0)
    sigma = z.std(dim=0)
        
    loss = sigma.sum()
    loss.backward()
    optimizer.step()

    print(f"iter {iter} loss {loss} sigma {sigma.sum().item()}")
    
path = "optimize/PFE/"
optimized_image = torch.sigmoid(image).detach()
plt.imshow(optimized_image.cpu().numpy().squeeze(), cmap="gray")
plt.savefig(path + f"{idx}_laml_optimized.png")