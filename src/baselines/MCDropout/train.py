import torch.optim as optim
from pytorch_metric_learning import distances, losses, miners
from src.baselines.MCDropout.config import parse_args
from src.baselines.MCDropout.models import CIFAR10_MCDropout, MNIST_MCDropout, CUB200_MCDropout
from src.data_modules import CIFAR10DataModule, FashionMNISTDataModule, MNISTDataModule, CUB200DataModule
from src.lightning_modules.MCDropoutLightningModule import MCDropoutLightningModule


def run(args):
    args.gpu_id = [int(item) for item in args.gpu_id]

    if args.dataset == "MNIST":
        model = MNIST_MCDropout(args.embedding_size, args.linear)
        data_module = MNISTDataModule
    elif args.dataset == "CIFAR10":
        model = CIFAR10_MCDropout(args.embedding_size, args.linear)
        data_module = CIFAR10DataModule
    elif args.dataset == "Casia":
        raise NotImplementedError("Casia dataset is not implemented")
    elif args.dataset == "FashionMNIST":
        model = MNIST_MCDropout(args.embedding_size, args.linear)
        data_module = FashionMNISTDataModule
    elif args.dataset == "CUB200":
        model = CUB200_MCDropout(args.embedding_size, args.linear)
        data_module = CUB200DataModule

    data_module = data_module(
        args.data_dir,
        args.batch_size,
        args.num_workers,
        npos=1,
        nneg=5,
    )

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    loss = losses.ContrastiveLoss(
        collect_stats=True,
        distance=distances.LpDistance(normalize_embeddings=False, p=2, power=1),
    )

    miner = miners.BatchEasyHardMiner(
        pos_strategy="all",
        neg_strategy="all",
        distance=distances.LpDistance(normalize_embeddings=False, p=2, power=1),
    )

    trainer = MCDropoutLightningModule(
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
