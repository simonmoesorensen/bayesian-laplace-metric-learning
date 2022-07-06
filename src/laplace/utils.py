import sys

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


def generate_predictions_from_samples(loader, weight_samples, full_model, inference_model=None, device="cpu"):
    if inference_model is None:
        inference_model = full_model

    preds = []
    for net_sample in weight_samples:
        vector_to_parameters(net_sample, inference_model.parameters())
        sample_preds = []
        for x, _ in iter(loader):
            x = x.to(device)
            pred = full_model(x)
            sample_preds.append(pred)
        preds.append(torch.cat(sample_preds, dim=0))
    return torch.stack(preds, dim=0)


def generate_predictions_from_samples_rolling(loader, weight_samples, full_model, inference_model=None, device="cpu"):
    """
    Welford's online algorithm for calculating mean and variance.
    """
    if inference_model is None:
        inference_model = full_model

    def get_single_sample_pred(sample) -> torch.Tensor:
        vector_to_parameters(sample, inference_model.parameters())
        sample_preds = []
        for i, (x, _) in enumerate(iter(loader)):
            # if i == 20:
            #     break
            x = x.to(device)
            pred = full_model(x)
            sample_preds.append(pred)
        return torch.cat(sample_preds, dim=0)

    N = len(weight_samples)

    mean = get_single_sample_pred(weight_samples[0, :])
    msq = 0.0
    delta = 0.0

    for i, net_sample in enumerate(weight_samples[1:, :]):
        sample_preds = get_single_sample_pred(net_sample)
        delta = sample_preds - mean
        mean += delta / (i + 1)
        msq += delta * (sample_preds - mean)

    variance = msq / (N - 1)
    return mean, variance


# def generate_predictions_from_samples_rolling(loader, weight_samples, full_model, inference_model=None, device="cpu"):
#     if inference_model is None:
#         inference_model = full_model

#     mean = 0
#     square_mean = 0
#     for net_sample in weight_samples:
#         vector_to_parameters(net_sample, inference_model.parameters())
#         sample_preds = []
#         for x, _ in iter(loader):
#             x = x.to(device)
#             pred = full_model(x)
#             sample_preds.append(pred)
#         sample_preds = torch.cat(sample_preds, dim=0)
#         mean += sample_preds
#         square_mean += sample_preds**2
#     mean = mean / len(weight_samples)
#     square_mean = square_mean / len(weight_samples)
#     variance = square_mean - mean**2
#     return mean, variance


# def generate_predictions_from_samples(loader, weight_samples, full_model, inference_model=None, device="cpu"):
#     if inference_model is None:
#         inference_model = full_model

#     means = []
#     vars = []
#     for step, (x, _) in enumerate(iter(loader)):
#         if step == 26:
#             break
#         x = x.to(device)
#         sample_preds = []
#         for net_sample in weight_samples:
#             vector_to_parameters(net_sample, inference_model.parameters())
#             sample_preds.append(full_model(x))
#         preds = torch.stack(sample_preds, dim=0)
#         means.append(preds.mean(dim=0))
#         vars.append(preds.var(dim=0))
#     return torch.cat(means, dim=0), torch.cat(vars, dim=0)


def generate_fake_predictions_from_samples(loader, weight_samples, full_model, inference_model=None, device="cpu"):
    if inference_model is None:
        inference_model = full_model

    preds = []
    for net_sample in weight_samples:
        vector_to_parameters(net_sample, inference_model.parameters())
        sample_preds = []
        for step, (x, _) in enumerate(iter(loader)):
            if step == 26:
                break
            x = torch.randn(x.shape, device=device)
            pred = full_model(x)
            sample_preds.append(pred)
        preds.append(torch.cat(sample_preds, dim=0))
    return torch.stack(preds, dim=0)


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
