import torch.optim as optim
from pytorch_metric_learning import distances, losses, miners
from src.baselines.MCDropout.config import parse_args
from src.baselines.MCDropout.models import CIFAR10_MCDropout, MNIST_MCDropout, CUB200ConvNet
from src.data_modules import CIFAR10DataModule, FashionMNISTDataModule, MNISTDataModule, CUB200DataModule
from src.lightning_modules.MCDropoutLightningModule import MCDropoutLightningModule


def run(MCDropout_args):
    MCDropout_args.gpu_id = [int(item) for item in MCDropout_args.gpu_id]

    if MCDropout_args.dataset == "MNIST":
        model = MNIST_MCDropout(embedding_size=MCDropout_args.embedding_size)
        data_module = MNISTDataModule
    elif MCDropout_args.dataset == "CIFAR10":
        model = CIFAR10_MCDropout(embedding_size=MCDropout_args.embedding_size)
        data_module = CIFAR10DataModule
    elif MCDropout_args.dataset == "Casia":
        raise NotImplementedError("Casia dataset is not implemented")
    elif MCDropout_args.dataset == "FashionMNIST":
        model = MNIST_MCDropout(embedding_size=MCDropout_args.embedding_size)
        data_module = FashionMNISTDataModule
    elif args.dataset == "CUB200":
        model = CUB200ConvNet(latent_dim=args.embedding_size)
        data_module = CUB200DataModule

    data_module = data_module(
        MCDropout_args.data_dir,
        MCDropout_args.batch_size,
        MCDropout_args.num_workers,
        npos=1,
        nneg=5,
    )

    optimizer = optim.Adam(model.parameters(), lr=MCDropout_args.lr)

    loss = losses.ContrastiveLoss(
        distance=distances.LpDistance(normalize_embeddings=False, p=2, power=1),
    )

    miner = miners.BatchEasyHardMiner(
        pos_strategy="all",
        neg_strategy="all",
        distance=distances.LpDistance(normalize_embeddings=False, p=2, power=1),
    )

    trainer = MCDropoutLightningModule(
        accelerator="gpu", devices=len(MCDropout_args.gpu_id), strategy="dp"
    )

    trainer.init(
        model=model, loss_fn=loss, miner=miner, optimizer=optimizer, args=MCDropout_args
    )

    trainer.add_data_module(data_module)

    trainer.train()
    trainer.test()
    trainer.log_hyperparams()
    trainer.save_model(prefix="Final")

    return trainer


if __name__ == "__main__":
    dul_train = run(parse_args())
