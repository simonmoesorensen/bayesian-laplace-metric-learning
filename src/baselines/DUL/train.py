import torch.optim as optim
from pytorch_metric_learning import distances, losses, miners
from src.baselines.DUL.config import parse_args
from src.baselines.DUL.models import CIFAR10_DUL, MNIST_DUL, Casia_DUL, Cub200_DUL
from src.data_modules import (
    CasiaDataModule,
    CIFAR10DataModule,
    FashionMNISTDataModule,
    MNISTDataModule,
    CUB200DataModule,
)
from src.lightning_modules.DULLightningModule import DULLightningModule
from src.utils import separate_batchnorm_params


def run(args):
    args.gpu_id = [int(item) for item in args.gpu_id]

    if args.dataset == "MNIST":
        model = MNIST_DUL(args.embedding_size, args.linear)
        data_module = MNISTDataModule
    elif args.dataset == "CIFAR10":
        model = CIFAR10_DUL(args.embedding_size, args.linear)
        data_module = CIFAR10DataModule
    elif args.dataset == "Casia":
        model = Casia_DUL(args.embedding_size, args.linear)
        data_module = CasiaDataModule
    elif args.dataset == "FashionMNIST":
        model = MNIST_DUL(args.embedding_size, args.linear)
        data_module = FashionMNISTDataModule
    elif args.dataset == "CUB200":
        model = Cub200_DUL(args.embedding_size, args.linear)
        data_module = CUB200DataModule

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

    loss = losses.NormalizedSoftmaxLoss(
        collect_stats=True,
        num_classes=data_module.n_classes,
        embedding_size=args.embedding_size,
    )

    miner = miners.BatchEasyHardMiner(
        pos_strategy="all",
        neg_strategy="all",
        distance=distances.LpDistance(normalize_embeddings=False, p=2, power=1),
    )

    trainer = DULLightningModule(
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
    dul_train = run(parse_args())
