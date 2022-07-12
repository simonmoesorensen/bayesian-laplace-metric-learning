from pytorch_metric_learning import testers
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
import faiss
from torch.utils.data import Subset


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
    accuracy_calculator = AccuracyCalculator(include=("mean_average_precision", "precision_at_1"), k=50)

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


def get_embedding_indices_within_margin(z_anchor, margin, model, train_set, device):
    latent_size = z_anchor.shape[-1]
    z = get_all_embeddings(train_set, model, device)[0]

    if device == "cpu":
        index = faiss.IndexFlatL2(latent_size)
    else:
        cfg = faiss.GpuIndexFlatConfig()
        cfg.useFloat16 = False
        cfg.device = 0

        flat_config = cfg
        resources = faiss.StandardGpuResources()
        index = faiss.GpuIndexFlatL2(resources, latent_size, flat_config)

    index.add(z)

    distances, indices = index.search(z_anchor, k=1000)
    indices_within_margin = indices[distances < margin]

    if isinstance(train_set, Subset):
        return train_set.indices[indices_within_margin]

    return indices_within_margin
