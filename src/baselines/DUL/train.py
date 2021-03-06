from src.lightning_modules.DULLightningModule import DULLightningModule
import torch.optim as optim
from pytorch_metric_learning import losses, miners

from src.data_modules.MNISTDataModule import MNISTDataModule
from src.data_modules.CIFAR10DataModule import CIFAR10DataModule
from src.data_modules.CasiaDataModule import CasiaDataModule

from src.baselines.DUL.config import parse_args
from src.baselines.DUL.models import MNIST_DUL, CIFAR10_DUL, Casia_DUL

from src.utils import (
    separate_batchnorm_params,
)


def run(dul_args):
    dul_args.gpu_id = [int(item) for item in dul_args.gpu_id]
    
    if dul_args.dataset == "MNIST":
        model = MNIST_DUL(embedding_size=dul_args.embedding_size)
        data_module = MNISTDataModule
    elif dul_args.dataset == "CIFAR10":
        model = CIFAR10_DUL(embedding_size=dul_args.embedding_size)
        data_module = CIFAR10DataModule
    elif dul_args.dataset == "Casia":
        model = Casia_DUL(embedding_size=dul_args.embedding_size)
        data_module = CasiaDataModule

    data_module = data_module(
        dul_args.data_dir, dul_args.batch_size, dul_args.num_workers
    )

    # Don't apply weight decay to batchnorm layers
    params_w_bn, params_no_bn = separate_batchnorm_params(model)

    optimizer = optim.Adam(
        [
            {
                "params": params_no_bn,
                "weight_decay": dul_args.weight_decay,
            },
            {"params": params_w_bn},
        ],
        lr=dul_args.lr,
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    loss = losses.ArcFaceLoss(
        num_classes=data_module.n_classes,
        scale=dul_args.arcface_scale,
        margin=dul_args.arcface_margin,
        embedding_size=dul_args.embedding_size,
    )

    # miner = miners.TripletMarginMiner(
    #     margin=dul_args.triplet_margin,
    # )

    miner = miners.MultiSimilarityMiner(
        epsilon=0.1,
    )

    trainer = DULLightningModule(
        accelerator="gpu", 
        devices=len(dul_args.gpu_id), 
        strategy="dp")

    trainer.init(
        model=model,
        loss_fn=loss,
        miner=miner,
        optimizer=optimizer,
        args=dul_args
    )

    trainer.add_data_module(data_module)

    trainer.train()
    trainer.test()
    trainer.log_hyperparams()
    trainer.save_model(prefix='Final')

    return trainer


if __name__ == "__main__":
    dul_train = run(parse_args())
