import torch

from src.baselines.models import CIFAR10ConvNet, FashionMNISTConvNet,FashionMNISTLinearNet, CIFAR10LinearNet
from src.lightning_modules.PostHocLaplaceLightningModule import (
    PostHocLaplaceLightningModule,
)
from src.baselines.Laplace_posthoc.config import parse_args

from src.data_modules import (
    CIFAR10DataModule,
    FashionMNISTDataModule,
    MNISTDataModule,
    CUB200DataModule,
)
from src.laplace.hessian.layerwise import (
    ContrastiveHessianCalculator,
    FixedContrastiveHessianCalculator,
)
from src.miners import AllCombinationsMiner, AllPositiveMiner


def run(args):
    args.gpu_id = [int(item) for item in args.gpu_id]
    torch.manual_seed(args.random_seed)
    
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
    
    
    if args.hessian == "positives":
        calculator_cls = ContrastiveHessianCalculator
        miner = AllPositiveMiner()
        n_neg = 0
    elif args.hessian == "fixed":
        calculator_cls = FixedContrastiveHessianCalculator
        miner = AllCombinationsMiner()
        n_neg = 1
    elif args.hessian == "full":
        calculator_cls = ContrastiveHessianCalculator
        miner = AllCombinationsMiner()
        n_neg = 1
    else:
        raise ValueError(f"Unknown method: {args.hessian}")

    data_module = data_module(
        args.data_dir,
        args.batch_size,
        args.num_workers,
        npos=1,
        nneg=n_neg,
    )



    trainer = PostHocLaplaceLightningModule(
        accelerator="gpu", devices=len(args.gpu_id), strategy="dp"
    )

    data_module.setup()
    dataset_size = data_module.dataset_train.__len__()
    trainer.init(
        model=model,
        miner=miner,
        calculator_cls=calculator_cls,
        dataset_size=dataset_size,
        args=args,
    )

    trainer.add_data_module(data_module)

    trainer.train()
    trainer.test()
    trainer.log_hyperparams()
    trainer.save_model(prefix="Final")

    return trainer


if __name__ == "__main__":
    run(parse_args())
