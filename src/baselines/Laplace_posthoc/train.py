import torch

from pytorch_metric_learning import distances, miners, losses
from src.baselines.models import CIFAR10ConvNet, FashionMNISTConvNet
from src.lightning_modules.PostHocLaplaceLightningModule import (
    PostHocLaplaceLightningModule,
)
from src.baselines.Laplace_posthoc.config import parse_args

from src.data_modules import (
    CIFAR10DataModule,
    FashionMNISTDataModule,
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
        raise NotImplementedError()
    elif args.dataset == "CIFAR10":
        model = CIFAR10ConvNet(latent_dim=args.embedding_size)
        data_module = CIFAR10DataModule
    elif args.dataset == "Casia":
        raise NotImplementedError()
    elif args.dataset == "FashionMNIST":
        model = FashionMNISTConvNet(latent_dim=args.embedding_size)
        data_module = FashionMNISTDataModule
    else:
        raise ValueError("Dataset not supported")

    data_module = data_module(
        args.data_dir,
        args.batch_size,
        args.num_workers,
        npos=1,
        nneg=1,
    )

    if args.hessian == "positives":
        calculator_cls = ContrastiveHessianCalculator
        miner = AllPositiveMiner()
    elif args.hessian == "fixed":
        calculator_cls = FixedContrastiveHessianCalculator
        miner = AllCombinationsMiner()
    elif args.hessian == "full":
        calculator_cls = ContrastiveHessianCalculator
        miner = AllCombinationsMiner()
    else:
        raise ValueError(f"Unknown method: {args.hessian}")

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
