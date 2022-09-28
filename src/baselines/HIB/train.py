import torch.optim as optim
from src.baselines.HIB.config import parse_args
from src.baselines.HIB.losses import SoftContrastiveLoss
from src.baselines.HIB.models import CIFAR10_HIB, MNIST_HIB, Casia_HIB, CUB200ConvNet
from src.data_modules import (
    CasiaDataModule,
    CIFAR10DataModule,
    FashionMNISTDataModule,
    MNISTDataModule,
    CUB200DataModule,
)
from src.lightning_modules.HIBLightningModule import HIBLightningModule
from src.utils import separate_batchnorm_params


def run(args):
    args.gpu_id = [int(item) for item in args.gpu_id]

    if args.dataset == "MNIST":
        model = MNIST_HIB(embedding_size=args.embedding_size)
        data_module = MNISTDataModule
    elif args.dataset == "CIFAR10":
        model = CIFAR10_HIB(embedding_size=args.embedding_size)
        data_module = CIFAR10DataModule
    elif args.dataset == "Casia":
        model = Casia_HIB(embedding_size=args.embedding_size)
        data_module = CasiaDataModule
    elif args.dataset == "FashionMNIST":
        model = MNIST_HIB(embedding_size=args.embedding_size)
        data_module = FashionMNISTDataModule
    elif args.dataset == "CUB200":
        model = CUB200ConvNet(latent_dim=args.embedding_size)
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
    )

    loss = SoftContrastiveLoss()

    # HIB uses own sampling method
    miner = None

    trainer = HIBLightningModule(
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
