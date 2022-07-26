from pytorch_metric_learning.losses import ContrastiveLoss
from pytorch_metric_learning import miners
from torch import optim

from src.laplace.models import MNIST, CIFAR10, Casia
from src.data_modules.MNISTDataModule import MNISTDataModule
from src.data_modules.CIFAR10DataModule import CIFAR10DataModule
from src.data_modules.CasiaDataModule import CasiaDataModule
from src.lightning_modules.PostHocLaplaceLightningModule import (
    PostHocLaplaceLightningModule,
)
from src.lightning_modules.BackboneLightningModule import BackboneLightningModule
from src.miners import AllPermutationsMiner
from src.laplace.PostHoc.config import parse_args


def run(args):
    args.gpu_id = [int(item) for item in args.gpu_id]

    sampler = None
    if args.dataset == "MNIST":
        model = MNIST(embedding_size=args.embedding_size)
        data_module = MNISTDataModule
    elif args.dataset == "CIFAR10":
        model = CIFAR10(embedding_size=args.embedding_size)
        data_module = CIFAR10DataModule
    elif args.dataset == "Casia":
        model = Casia(embedding_size=args.embedding_size)
        data_module = CasiaDataModule
        sampler = "WeightedRandomSampler"
    else:
        raise ValueError("Unknown dataset: {}".format(args.dataset))

    data_module = data_module(
        args.data_dir,
        args.batch_size,
        args.num_workers,
        shuffle=args.shuffle,
        pin_memory=args.pin_memory,
        sampler=sampler,
    )

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    loss = ContrastiveLoss(neg_margin=args.neg_margin)

    # Find MAP solution
    miner = miners.MultiSimilarityMiner()
    map_trainer = BackboneLightningModule()
    map_trainer.init(model, loss, miner, optimizer, args)
    map_trainer.add_data_module(data_module)
    if not args.model_path:
        map_trainer.train()
    map_trainer.test()

    # Post-hoc
    miner = AllPermutationsMiner()
    post_hoc_trainer = PostHocLaplaceLightningModule(
        accelerator="gpu", devices=len(args.gpu_id), strategy="dp"
    )
    post_hoc_trainer.init(model, loss, miner, optimizer, args)
    post_hoc_trainer.add_data_module(data_module)
    post_hoc_trainer.train()
    post_hoc_trainer.test()
    post_hoc_trainer.log_hyperparams()
    post_hoc_trainer.save_model(prefix="Final")

    return post_hoc_trainer


if __name__ == "__main__":
    run(parse_args())
