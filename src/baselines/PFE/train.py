from src.lightning_modules.PFELightningModule import PFELightningModule
import torch.optim as optim
from pytorch_metric_learning import miners, distances

from src.data_modules import (
    FashionMNISTDataModule,
    MNISTDataModule,
    CIFAR10DataModule,
    CasiaDataModule,
)
from src.baselines.PFE.config import parse_args
from src.baselines.PFE.models import (
    MNIST_PFE,
    CIFAR10_PFE,
    Casia_PFE,
    FashionMNIST_PFE,
)
from src.utils import (
    separate_batchnorm_params,
)

from src.baselines.PFE.losses import MLSLoss


def run(PFE_args):
    PFE_args.gpu_id = [int(item) for item in PFE_args.gpu_id]

    sampler = None

    if PFE_args.dataset == "MNIST":
        model = MNIST_PFE(embedding_size=PFE_args.embedding_size)
        data_module = MNISTDataModule
    elif PFE_args.dataset == "CIFAR10":
        model = CIFAR10_PFE(embedding_size=PFE_args.embedding_size)
        data_module = CIFAR10DataModule
    elif PFE_args.dataset == "Casia":
        model = Casia_PFE(embedding_size=PFE_args.embedding_size)
        data_module = CasiaDataModule
        sampler = "WeightedRandomSampler"
    elif PFE_args.dataset == "FashionMNIST":
        model = FashionMNIST_PFE(
            loss=PFE_args.loss, embedding_size=PFE_args.embedding_size
        )
        data_module = FashionMNISTDataModule

    data_module = data_module(
        PFE_args.data_dir,
        PFE_args.batch_size,
        PFE_args.num_workers,
        shuffle=PFE_args.shuffle,
        pin_memory=PFE_args.pin_memory,
        sampler=sampler,
    )

    # Don't apply weight decay to batchnorm layers
    params_w_bn, params_no_bn = separate_batchnorm_params(model)

    optimizer = optim.Adam(
        [
            {
                "params": params_no_bn,
                "weight_decay": PFE_args.weight_decay,
            },
            {"params": params_w_bn},
            # Add beta and gamma as learnable parameters to the optimizer
            {"params": model.beta},
            {"params": model.gamma},
        ],
        lr=PFE_args.lr,
    )

    loss = MLSLoss()

    # Get all positive pairs for the loss
    miner = miners.BatchEasyHardMiner(
        pos_strategy="all",
        neg_strategy="hard",
        distance=distances.LpDistance(normalize_embeddings=False, p=2, power=1),
    )

    trainer = PFELightningModule(
        accelerator="gpu", devices=len(PFE_args.gpu_id), strategy="dp"
    )

    trainer.init(
        model=model, loss_fn=loss, miner=miner, optimizer=optimizer, args=PFE_args
    )

    trainer.add_data_module(data_module)

    trainer.train()
    trainer.test()
    trainer.log_hyperparams()
    trainer.save_model(prefix="Final")

    return trainer


if __name__ == "__main__":
    dul_train = run(parse_args())
