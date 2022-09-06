import torch
from src.laplace.models import CIFAR10_Laplace, FashionMNIST_Laplace
from src.lightning_modules.PostHocLaplaceLightningModule import (
    PostHocLaplaceLightningModule,
)
from src.laplace.config import parse_args

from src.data_modules import (
    CIFAR10DataModule,
    FashionMNISTDataModule,
)
from src.hessian.layerwise import (
    ContrastiveHessianCalculator,
    FixedContrastiveHessianCalculator,
)
from src.miners import AllCombinationsMiner, AllPositiveMiner


def run(args):
    args.gpu_id = [int(item) for item in args.gpu_id]
    torch.manual_seed(args.random_seed)

    sampler = None

    if args.dataset == "MNIST":
        raise NotImplementedError()
    elif args.dataset == "CIFAR10":
        model = CIFAR10_Laplace(embedding_size=args.embedding_size)
        data_module = CIFAR10DataModule
    elif args.dataset == "Casia":
        raise NotImplementedError()
    elif args.dataset == "FashionMNIST":
        model = FashionMNIST_Laplace(embedding_size=args.embedding_size)
        data_module = FashionMNISTDataModule
    else:
        raise ValueError("Dataset not supported")

    data_module = data_module(
        args.data_dir,
        args.batch_size,
        args.num_workers,
        shuffle=args.shuffle,
        pin_memory=args.pin_memory,
        sampler=sampler,
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

    inference_model = getattr(model.backbone, args.inference_model)

    trainer.init(
        model=model,
        miner=miner,
        calculator_cls=calculator_cls,
        inference_model=inference_model,
        args=args,
    )

    trainer.add_data_module(data_module)

    trainer.train()
    trainer.test(expected=True)
    trainer.log_hyperparams()
    trainer.save_model(prefix="Final")

    return trainer


if __name__ == "__main__":
    run(parse_args())
