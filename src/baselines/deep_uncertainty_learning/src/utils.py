import torch
import torch.nn as nn

from pytorch_metric_learning import testers
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator


def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)

    return output


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_all_embeddings(dataset, model, data_device, batch_size, num_workers):
    """
    Wrapper function to get embeddings from BaseTester.
    """
    # Dataloader_num_workers has to be 0 to avoid pid error
    # This only happens when within multiprocessing
    class Embedder(nn.Module):
        def __init__(self):
            super(Embedder, self).__init__()
            self.model = nn.Identity()

        def forward(self, x):
            """
            Generate embeddings from the normal distribution
            """
            mu, std = self.model(x)

            epsilon = torch.randn_like(std)
            embeddings = mu + epsilon * std
            return embeddings

    embedder_model = Embedder()

    tester = testers.BaseTester(
        dataloader_num_workers=num_workers,
        data_device=data_device,
        batch_size=batch_size,
    )

    return tester.get_all_embeddings(dataset, model, embedder_model)


def separate_batchnorm_params(modules):
    if not isinstance(modules, list):
        modules = [*modules.modules()]
    paras_only_bn = []
    paras_wo_bn = []
    for layer in modules:
        if "model" in str(layer.__class__):
            continue
        if "container" in str(layer.__class__):
            continue
        else:
            if "batchnorm" in str(layer.__class__):
                paras_only_bn.extend([*layer.parameters()])
            else:
                paras_wo_bn.extend([*layer.parameters()])

    return paras_only_bn, paras_wo_bn


def test_model(train_set, test_set, model, data_device, batch_size, num_workers):
    """
    Compute accuracy using AccuracyCalculator from pytorch-metric-learning
    """
    accuracy_calculator = AccuracyCalculator(
        include=("mean_average_precision", "precision_at_1"), k=50
    )

    train_embeddings, train_labels = get_all_embeddings(
        train_set, model, data_device, batch_size, num_workers
    )
    test_embeddings, test_labels = get_all_embeddings(
        test_set, model, data_device, batch_size, num_workers
    )

    accuracies = accuracy_calculator.get_accuracy(
        test_embeddings,
        train_embeddings,
        test_labels.squeeze(),
        train_labels.squeeze(),
        embeddings_come_from_same_source=True,
    )
    return accuracies
