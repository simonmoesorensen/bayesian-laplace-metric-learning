import torch.optim as optim
from pytorch_metric_learning import distances, miners
from src.baselines.PFE.config import parse_args
from src.baselines.PFE.losses import MLSLoss
from src.baselines.PFE.models import CIFAR10_PFE, MNIST_PFE, Casia_PFE, FashionMNIST_PFE
from src.data_modules import (
    CasiaDataModule,
    CIFAR10DataModule,
    FashionMNISTDataModule,
    MNISTDataModule,
)
from src.lightning_modules.PFELightningModule import PFELightningModule
from src.utils import separate_batchnorm_params
import torch

def run(args):
    args.gpu_id = [int(item) for item in args.gpu_id]
    
    if args.dataset == "MNIST":
        model = MNIST_PFE(
            embedding_size=args.embedding_size, seed=args.random_seed, linear=args.linear
        )
        data_module = MNISTDataModule
    elif args.dataset == "CIFAR10":
        model = CIFAR10_PFE(
            embedding_size=args.embedding_size, seed=args.random_seed, linear=args.linear
        )
        data_module = CIFAR10DataModule
    elif args.dataset == "Casia":
        model = Casia_PFE(
            embedding_size=args.embedding_size, seed=args.random_seed, linear=args.linear
        )
        data_module = CasiaDataModule
    elif args.dataset == "FashionMNIST":
        model = FashionMNIST_PFE(
            embedding_size=args.embedding_size, seed=args.random_seed, linear=args.linear
        )
        data_module = FashionMNISTDataModule

    # do not use negative in training.
    data_module = data_module(
        args.data_dir,
        args.batch_size,
        args.num_workers,
        npos=1,
        nneg=0,
    )

    # Don't apply weight decay to batchnorm layers
    params_w_bn, params_no_bn = separate_batchnorm_params(model)

    optimizer = optim.Adam(
        [
            {
                "params": params_no_bn,
                "weight_decay": args.weight_decay,
            },
            {"params": params_w_bn},
            # Add beta and gamma as learnable parameters to the optimizer
            {"params": model.beta},
            {"params": model.gamma},
        ],
        lr=args.lr,
    )

    loss = MLSLoss()

    miner = miners.BatchEasyHardMiner(
        pos_strategy="all",
        neg_strategy="all",
        distance=distances.LpDistance(normalize_embeddings=False, p=2, power=1),
    )

    trainer = PFELightningModule(
        accelerator="gpu", devices=len(args.gpu_id), strategy="dp"
    )

    trainer.init(
        model=model, loss_fn=loss, miner=miner, optimizer=optimizer, args=args
    )

    trainer.add_data_module(data_module)

    trainer.train()
    trainer.test()
    trainer.log_hyperparams()
    trainer.save_model(prefix="Final")
    return trainer


if __name__ == "__main__":
    dul_train = run(parse_args())
