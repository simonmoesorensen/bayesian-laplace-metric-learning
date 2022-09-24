
from torchvision import transforms
import torchvision.datasets as d

import os, sys

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
from pytorch_metric_learning import losses, miners
from dotmap import DotMap


args = {"latent_dim": 3,
        "data_dir": "/work3/frwa/datasets/",
        "dataset": "FashionMNIST",
        "batch_size": 32,
        "num_workers": 8,
        "gpu_id": [0],
        "model": "Posthoc",
        "random_seed": 42,
        "log_dir": "",
        "vis_dir": ""}

if args["model"] == "Backbone":
    args["model_path"] = "outputs/Backbone/checkpoints/FashionMNIST/latent_dim_3_seed_42_conv/Model_Epoch_60_Time_2022-09-24T110808_checkpoint.pth"
    module = BackboneLightningModule
elif args["model"] == "PFE":
    args["model_path"] = "outputs/PFE/checkpoints/FashionMNIST/latent_dim_3_seed_42_conv/Final_Model_Epoch_200_Time_2022-09-24T120547_checkpoint.pth"
    module = PFELightningModule
elif args["model"] == "Posthoc":
    args["model_path"] = "outputs/Laplace_posthoc/checkpoints/FashionMNIST/latent_dim_3_seed_42_conv/Final_Model_Epoch_1_Time_2022-09-24T115046_checkpoint.pth"
    module = PostHocLaplaceLightningModule
elif args["model"] == "Online":
    args["model_path"] = "outputs/Laplace_online/checkpoints/FashionMNIST/latent_dim_3_seed_42_mem_0_999_conv/Final_Model_Epoch_300_Time_2022-09-24T134047_checkpoint.pth"
    module = PostHocLaplaceLightningModule
else:
    raise ValueError("Model not supported")

args = DotMap(args)

if args.model == "PFE":
    model = FashionMNIST_PFE(embedding_size=args.latent_dim)
else:
    model = FashionMNISTConvNet(latent_dim=args.latent_dim)
    
model.load_state_dict(torch.load(args.model_path))
    
data_module = FashionMNISTDataModule
miner = None

data_module = data_module(
    args.data_dir,
    args.batch_size,
    args.num_workers,
    npos=1,
    nneg=0,
)

trainer = module(
    accelerator="gpu", devices=len(args.gpu_id), strategy="dp"
)

trainer.model = model.to("cuda:0")
trainer.add_data_module(data_module)
trainer.n_test_samples = 100

if args.model in ("Posthoc", "Online"):
    if args.model == "Posthoc":
        path = "outputs/Laplace_posthoc/checkpoints/FashionMNIST/latent_dim_3_seed_42_conv/"
        #trainer.scale = torch.load(path + "scale.pth").to("cuda:0")
        #trainer.prior_prec = torch.load(path + "pror_prec.pth").to("cuda:0")
        trainer.scale = torch.tensor(1.0).to("cuda:0")
        trainer.prior_prec = torch.tensor(1.0).to("cuda:0")
    else:
        path = "outputs/Laplace_online/checkpoints/FashionMNIST/latent_dim_3_seed_42_mem_0_999_conv/"
        trainer.scale = torch.tensor(1.0).to("cuda:0")
        trainer.prior_prec = torch.tensor(1.0).to("cuda:0")
    trainer.hessian = torch.load(path + "hessian.pth").to("cuda:0")


z_mu, z_sigma, z_samples, labels, images = trainer.compute_features(trainer.test_loader, n_samples=trainer.n_test_samples)
ood_z_mu, ood_z_sigma, ood_z_samples, ood_labels, ood_images  = trainer.compute_features(trainer.ood_loader, n_samples=trainer.n_test_samples)


path = f"notebooks/outputs/{args.model}/{args.dataset}/latent_dim_{args.latent_dim}/"
os.makedirs(path, exist_ok=True)
torch.save(z_mu, f"{path}/z_mu.pt")
torch.save(labels, f"{path}/labels.pt")
if z_sigma is not None and len(z_sigma) > 0:
    torch.save(z_sigma, f"{path}/z_sigma.pt")
    torch.save(z_samples, f"{path}/z_samples.pt")
    
torch.save(ood_z_mu, f"{path}/ood_z_mu.pt")
torch.save(ood_labels, f"{path}/ood_labels.pt")
if z_sigma is not None and len(z_sigma) > 0:
    torch.save(ood_z_sigma, f"{path}/ood_z_sigma.pt")
    torch.save(ood_z_samples, f"{path}/ood_z_samples.pt")