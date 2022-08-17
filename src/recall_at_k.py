import torch
from pytorch_metric_learning.utils import accuracy_calculator


class AccuracyRecall(accuracy_calculator.AccuracyCalculator):
    def calculate_recall_at_k(
        self,
        query,
        reference,
        query_labels,
        reference_labels,
        embeddings_come_from_same_source,
        label_comparison_fn,
        label_counts,
        knn_labels,
        knn_distances,
        lone_query_labels,
        not_lone_query_mask,
    ):
        mask = (
            knn_labels == query_labels[:, None]
        )  # see if label of point is in the labels of k nearest neighbors
        s = mask.sum(axis=1).gt(0)  # we don't care how many equal occurrences there are
        s = s.type(torch.DoubleTensor)
        return s.mean().item()

    def requires_knn(self):
        return super().requires_knn() + ["recall_at_k"]
