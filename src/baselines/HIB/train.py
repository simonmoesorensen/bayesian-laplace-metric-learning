from src.lightning_modules.HIBLightningModule import HIBLightningModule
import torch.optim as optim
from pytorch_metric_learning import miners

from src.data_modules.MNISTDataModule import MNISTDataModule
from src.data_modules.CIFAR10DataModule import CIFAR10DataModule
from src.data_modules.CasiaDataModule import CasiaDataModule

from src.baselines.HIB.config import parse_args
from src.baselines.HIB.models import MNIST_HIB, CIFAR10_HIB, Casia_HIB
from src.baselines.HIB.losses import SoftContrastiveLoss

from src.utils import (
    separate_batchnorm_params,
)


def run(HIB_args):
    HIB_args.gpu_id = [int(item) for item in HIB_args.gpu_id]

    sampler = None
    if HIB_args.dataset == "MNIST":
        model = MNIST_HIB(embedding_size=HIB_args.embedding_size)
        data_module = MNISTDataModule
    elif HIB_args.dataset == "CIFAR10":
        model = CIFAR10_HIB(embedding_size=HIB_args.embedding_size)
        data_module = CIFAR10DataModule
    elif HIB_args.dataset == "Casia":
        model = Casia_HIB(embedding_size=HIB_args.embedding_size)
        data_module = CasiaDataModule
        sampler = "WeightedRandomSampler"

    data_module = data_module(
        HIB_args.data_dir,
        HIB_args.batch_size,
        HIB_args.num_workers,
        shuffle=HIB_args.shuffle,
        pin_memory=HIB_args.pin_memory,
        sampler=sampler,
    )

    # Don't apply weight decay to batchnorm layers
    params_w_bn, params_no_bn = separate_batchnorm_params(model)

    optimizer = optim.Adam(
        [
            {
                "params": params_no_bn,
                "weight_decay": HIB_args.weight_decay,
            },
            {"params": params_w_bn},
        ],
        lr=HIB_args.lr,
    )

    loss = SoftContrastiveLoss()

    miner = miners.BatchEasyHardMiner(
        pos_strategy='all',
        neg_strategy='easy',
    )

    trainer = HIBLightningModule(
        accelerator="gpu", devices=len(HIB_args.gpu_id), strategy="dp"
    )

    trainer.init(
        model=model, loss_fn=loss, miner=miner, optimizer=optimizer, args=HIB_args
    )

    trainer.add_data_module(data_module)

    trainer.train()
    trainer.test()
    trainer.log_hyperparams()
    trainer.save_model(prefix="Final")

    return trainer


if __name__ == "__main__":
    dul_train = run(parse_args())
