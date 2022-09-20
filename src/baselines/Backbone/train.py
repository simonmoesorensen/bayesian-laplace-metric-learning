import torch.optim as optim
from pytorch_metric_learning import losses, miners
from src.baselines.Backbone.config import parse_args
from src.baselines.Backbone.models import (
    Casia_Backbone,
    CIFAR10_Backbone,
    MNIST_Backbone,
)
from src.data_modules import (
    CasiaDataModule,
    CIFAR10DataModule,
    FashionMNISTDataModule,
    MNISTDataModule,
)
from src.lightning_modules.BackboneLightningModule import BackboneLightningModule
from src.utils import separate_batchnorm_params


def run(Backbone_args):
    Backbone_args.gpu_id = [int(item) for item in Backbone_args.gpu_id]

    if Backbone_args.dataset == "MNIST":
        model = MNIST_Backbone(embedding_size=Backbone_args.embedding_size)
        data_module = MNISTDataModule
    elif Backbone_args.dataset == "CIFAR10":
        model = CIFAR10_Backbone(embedding_size=Backbone_args.embedding_size)
        data_module = CIFAR10DataModule
    elif Backbone_args.dataset == "Casia":
        model = Casia_Backbone(embedding_size=Backbone_args.embedding_size)
        data_module = CasiaDataModule
    elif Backbone_args.dataset == "FashionMNIST":
        model = MNIST_Backbone(embedding_size=Backbone_args.embedding_size)
        data_module = FashionMNISTDataModule
    else:
        raise ValueError("Dataset not supported")

    data_module = data_module(
        Backbone_args.data_dir,
        Backbone_args.batch_size,
        Backbone_args.num_workers,
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
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    loss = losses.ContrastiveLoss(
        # distance=distances.LpDistance(normalize_embeddings=False, p=2, power=1),
        # neg_margin=0.2, pos_margin=0.,
    )

    miner = miners.BatchEasyHardMiner(
        pos_strategy="all",
        neg_strategy="all",
        # distance=distances.LpDistance(normalize_embeddings=False, p=2, power=1),
    )

    trainer = BackboneLightningModule(
        accelerator="gpu", devices=len(Backbone_args.gpu_id), strategy="dp"
    )

    trainer.init(
        model=model, loss_fn=loss, miner=miner, optimizer=optimizer, args=Backbone_args
    )

    trainer.add_data_module(data_module)

    trainer.train()
    trainer.test(expected=False)
    trainer.log_hyperparams()
    trainer.save_model(prefix="Final")

    return trainer


if __name__ == "__main__":
    dul_train = run(parse_args())
