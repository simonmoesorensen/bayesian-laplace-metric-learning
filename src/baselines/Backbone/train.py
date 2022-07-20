from src.lightning_modules.BackboneLightningModule import BackboneLightningModule
import torch.optim as optim
from pytorch_metric_learning import miners, losses

from src.data_modules.MNISTDataModule import MNISTDataModule
from src.data_modules.CIFAR10DataModule import CIFAR10DataModule
from src.data_modules.CasiaDataModule import CasiaDataModule

from src.baselines.Backbone.config import parse_args
from src.baselines.Backbone.models import MNIST_Backbone, CIFAR10_Backbone, Casia_Backbone

from src.utils import (
    separate_batchnorm_params,
)


def run(Backbone_args):
    Backbone_args.gpu_id = [int(item) for item in Backbone_args.gpu_id]

    sampler = None

    if Backbone_args.dataset == "MNIST":
        model = MNIST_Backbone(embedding_size=Backbone_args.embedding_size)
        data_module = MNISTDataModule
    elif Backbone_args.dataset == "CIFAR10":
        model = CIFAR10_Backbone(embedding_size=Backbone_args.embedding_size)
        data_module = CIFAR10DataModule
    elif Backbone_args.dataset == "Casia":
        model = Casia_Backbone(embedding_size=Backbone_args.embedding_size)
        data_module = CasiaDataModule
        sampler = "WeightedRandomSampler"

    data_module = data_module(
        Backbone_args.data_dir,
        Backbone_args.batch_size,
        Backbone_args.num_workers,
        shuffle=Backbone_args.shuffle,
        pin_memory=Backbone_args.pin_memory,
        sampler=sampler,
    )

    # Don't apply weight decay to batchnorm layers
    params_w_bn, params_no_bn = separate_batchnorm_params(model)

    optimizer = optim.Adam(
        [
            {
                "params": params_no_bn,
                "weight_decay": Backbone_args.weight_decay,
            },
            {"params": params_w_bn},
        ],
        lr=Backbone_args.lr,
    )

    loss = losses.ContrastiveLoss()

    miner = miners.MultiSimilarityMiner()
    
    trainer = BackboneLightningModule(
        accelerator="gpu", devices=len(Backbone_args.gpu_id), strategy="dp"
    )

    trainer.init(
        model=model, loss_fn=loss, miner=miner, optimizer=optimizer, args=Backbone_args
    )

    trainer.add_data_module(data_module)

    trainer.train()
    trainer.test()
    trainer.log_hyperparams()
    trainer.save_model(prefix="Final")

    return trainer


if __name__ == "__main__":
    dul_train = run(parse_args())
