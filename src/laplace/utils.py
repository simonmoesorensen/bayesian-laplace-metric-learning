import torch
from pytorch_metric_learning import testers
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
from torch.nn.utils import vector_to_parameters


def get_all_embeddings(dataset, model, data_device):
    """
    Wrapper function to get embeddings from BaseTester.
    """
    # Dataloader_num_workers has to be 0 to avoid pid error
    # This only happens when within multiprocessing
    tester = testers.BaseTester(dataloader_num_workers=0, data_device=data_device)
    return tester.get_all_embeddings(dataset, model)


def test_model(train_set, test_set, model, data_device):
    """
    Compute accuracy using AccuracyCalculator from pytorch-metric-learning
    """
    accuracy_calculator = AccuracyCalculator(
        include=("mean_average_precision", "precision_at_1"), k=50
    )

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


def generate_predictions_from_samples(
    loader, weight_samples, full_model, inference_model=None, device="cpu"
):
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


def generate_fake_predictions_from_samples(
    loader, weight_samples, full_model, inference_model=None, device="cpu"
):
    if inference_model is None:
        inference_model = full_model

    preds = []
    for net_sample in weight_samples:
        vector_to_parameters(net_sample, inference_model.parameters())
        sample_preds = []
        for x, _ in iter(loader):
            x = torch.randn(x.shape, device=device)
            pred = full_model(x)
            sample_preds.append(pred)
        preds.append(torch.cat(sample_preds, dim=0))
    return torch.stack(preds, dim=0)


def get_sample_accuracy(train_set, test_set, model, inference_model, samples, device):
    accuracies = []
    for sample in samples:
        vector_to_parameters(sample, inference_model.parameters())
        accuracies.append(
            test_model(train_set, test_set, model, device)["precision_at_1"]
        )
    return accuracies


def sample_nn_weights(parameters, posterior_scale, n_samples=16):
    n_params = len(parameters)
    samples = torch.randn(n_samples, n_params, device=parameters.device)
    samples = samples * posterior_scale.reshape(1, n_params)
    return parameters.reshape(1, n_params) + samples
