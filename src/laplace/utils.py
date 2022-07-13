from typing import Tuple

import torch
from pytorch_metric_learning import testers
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
from torch.nn.utils.convert_parameters import vector_to_parameters


def get_all_embeddings(dataset, model, data_device):
    """
    Wrapper function to get embeddings from BaseTester.
    """
    # Dataloader_num_workers has to be 0 to avoid pid error
    # This only happens when within multiprocessing
    tester = testers.BaseTester(dataloader_num_workers=0, data_device=data_device)
    return tester.get_all_embeddings(dataset, model)


def test_model(train_set, test_set, model, data_device, k=10):
    """
    Compute accuracy using AccuracyCalculator from pytorch-metric-learning
    """
    accuracy_calculator = AccuracyCalculator(include=("mean_average_precision", "precision_at_1"), k=k)

    train_embeddings, train_labels = get_all_embeddings(train_set, model, data_device)
    test_embeddings, test_labels = get_all_embeddings(test_set, model, data_device)

    accuracies = accuracy_calculator.get_accuracy(
        test_embeddings,
        train_embeddings,
        test_labels.squeeze(),
        train_labels.squeeze(),
        embeddings_come_from_same_source=False,
    )
    return accuracies


def get_single_sample_pred(full_model, loader, device) -> torch.Tensor:
    preds = []
    for x, _ in iter(loader):
        with torch.inference_mode():
            pred = full_model(x.to(device))
        preds.append(pred)
    preds = torch.cat(preds, dim=0)
    return preds


def generate_predictions_from_samples_rolling(
    loader, weight_samples, full_model, inference_model=None, device="cpu"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Welford's online algorithm for calculating mean and variance.
    """
    if inference_model is None:
        inference_model = full_model

    N = len(weight_samples)

    vector_to_parameters(weight_samples[0, :], inference_model.parameters())
    mean = get_single_sample_pred(full_model, loader, device)
    msq = 0.0
    delta = 0.0

    for i, net_sample in enumerate(weight_samples[1:, :]):
        vector_to_parameters(net_sample, inference_model.parameters())
        sample_preds = get_single_sample_pred(full_model, loader, device)
        delta = sample_preds - mean
        mean += delta / (i + 1)
        msq += delta * delta

    variance = msq / (N - 1)
    return torch.tensor(mean), torch.tensor(variance)


def sample_normal(mean, variance, n_samples=16) -> torch.Tensor:
    assert variance.shape == mean.shape
    if len(mean.shape) == 1:
        n = mean.shape[0]
        new_shape = (n, n_samples)
        sample_dim = 1
        return torch.randn(new_shape, device=mean.device) * torch.sqrt(variance).unsqueeze(sample_dim).expand(
            new_shape
        ) + mean.unsqueeze(sample_dim).expand(new_shape)
    elif len(mean.shape) == 2:
        n, m = mean.shape
        new_shape = (n, m, n_samples)
        sample_dim = 2
        return torch.randn(new_shape, device=mean.device) * torch.sqrt(variance).unsqueeze(sample_dim).expand(
            new_shape
        ) + mean.unsqueeze(sample_dim).expand(new_shape)
    else:
        raise ValueError("Only 1D and 2D tensors are supported")


def get_sample_accuracy(train_set, test_set, model, inference_model, samples, device):
    accuracies = []
    for sample in samples:
        vector_to_parameters(sample, inference_model.parameters())
        accuracies.append(test_model(train_set, test_set, model, device)["precision_at_1"])
    return accuracies


def sample_nn_weights(parameters, posterior_scale, n_samples=16):
    n_params = len(parameters)
    samples = torch.randn(n_samples, n_params, device=parameters.device)
    samples = samples * posterior_scale.reshape(1, n_params)
    return parameters.reshape(1, n_params) + samples
