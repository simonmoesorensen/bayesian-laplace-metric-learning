from gc import collect
import torch.optim as optim
from pytorch_metric_learning import losses, miners
from src.baselines.Backbone.config import parse_args
from src.baselines.models import FashionMNISTLinearNet, FashionMNISTConvNet, CIFAR10ConvNet, CIFAR10LinearNet, CUB200ConvNet
from src.data_modules import (
    CIFAR10DataModule,
    FashionMNISTDataModule,
    MNISTDataModule,
    CUB200DataModule,
)
from src.lightning_modules.BackboneLightningModule import BackboneLightningModule
from src.utils import separate_batchnorm_params


def run(args):
    args.gpu_id = [int(item) for item in args.gpu_id]

    if args.dataset == "MNIST":
        if args.linear:
            model = FashionMNISTLinearNet(latent_dim=args.embedding_size)
        else:
            model = FashionMNISTConvNet(latent_dim=args.embedding_size)
        data_module = MNISTDataModule
    elif args.dataset == "CIFAR10":
        if args.linear:
            model = CIFAR10LinearNet(latent_dim=args.embedding_size)
        else:
            model = CIFAR10ConvNet(latent_dim=args.embedding_size)
        data_module = CIFAR10DataModule
    elif args.dataset == "FashionMNIST":
        if args.linear:
            model = FashionMNISTLinearNet(latent_dim=args.embedding_size)
        else:
            model = FashionMNISTConvNet(latent_dim=args.embedding_size)
        data_module = FashionMNISTDataModule
    elif args.dataset == "CUB200":
        model = CUB200ConvNet(latent_dim=args.embedding_size)
        data_module = CUB200DataModule
    else:
        raise ValueError("Dataset not supported")
    
    data_module = data_module(
        args.data_dir,
        args.batch_size,
        args.num_workers,
        npos=1,
        nneg=5,
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
        ],
        lr=args.lr,
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    loss = losses.ContrastiveLoss(
        collect_stats=True,
        # distance=distances.LpDistance(normalize_embeddings=False, p=2, power=1),
        # neg_margin=0.2, pos_margin=0.,
    )

    miner = miners.BatchEasyHardMiner(
        pos_strategy="all",
        neg_strategy="all",
        # distance=distances.LpDistance(normalize_embeddings=False, p=2, power=1),
    )

    trainer = BackboneLightningModule(
        accelerator="gpu", devices=len(args.gpu_id), strategy="dp"
    )

    trainer.init(
        model=model, loss_fn=loss, miner=miner, optimizer=optimizer, args=args
    )

    trainer.add_data_module(data_module)

    if args.train:
        trainer.train()
    trainer.test()
    trainer.log_hyperparams()
    trainer.save_model(prefix="Final")

    return trainer


if __name__ == "__main__":
    trainer = run(parse_args())
