from turtle import distance
from pytorch_metric_learning.utils import accuracy_calculator
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
from pytorch_metric_learning.utils.inference import CustomKNN
from pytorch_metric_learning.distances import LpDistance

from utils import get_all_embeddings
from src.distances import ExpectedSquareL2Distance


# class YourCalculator(accuracy_calculator.AccuracyCalculator):
#     def calculate_precision_at_1(self, knn_labels, query_labels, not_lone_query_mask, label_counts, **kwargs):
#         knn_labels, query_labels = accuracy_calculator.try_getting_not_lone_labels(
#             knn_labels, query_labels, not_lone_query_mask
#         )
#         if knn_labels is None:
#             return accuracy_calculator.zero_accuracy(label_counts[0], self.return_per_class)
#         return accuracy_calculator.precision_at_k(
#             knn_labels,
#             query_labels[:, None],
#             ,
#             self.avg_of_avgs,
#             self.return_per_class,
#             self.label_comparison_fn,
#         )

#     def calculate_fancy_mutual_info(self, query_labels, cluster_labels, **kwargs):
#         return accuracy_calculator.precision_at_k()

#     def requires_clustering(self):
#         return super().requires_clustering() + ["fancy_mutual_info"]

#     def requires_knn(self):
#         return super().requires_knn() + ["precision_at_2"]


class MetricsCalculator:
    def __init__(self, model, data_device, train_embeddings, train_labels, distance_metric=None) -> None:
        self.k_values = [1, 5, 10, 20]
        self.model = model
        self.data_device = data_device

        self.train_embeddings = train_embeddings
        self.train_labels = train_labels

        if distance_metric is None:
            distance_metric = LpDistance(p=2, power=2)
            # ExpectedSquareL2Distance(feature_dim=1, sample_dim=2)
        self.distance_metric = distance_metric

    def compute_metrics(self, test_embeddings, test_labels):
        metrics = {}
        for k in self.k_values:
            calculator = AccuracyCalculator(
                include=("mean_average_precision", "mean_average_precision_at_r", "r_precision", "precision_at_1"),
                k=k,
                # knn_func=CustomKNN(self.distance_metric),
            )
            accuracies = calculator.get_accuracy(
                test_embeddings,
                self.train_embeddings,
                test_labels.squeeze(),
                self.train_labels.squeeze(),
                embeddings_come_from_same_source=False,
            )
            metrics[f"mean_average_precision_at_r_{k}"] = accuracies["mean_average_precision_at_r"]
            metrics[f"r_precision_{k}"] = accuracies["mean_average_precision"]
            metrics[f"mean_average_precision_at_{k}"] = accuracies["mean_average_precision"]
        metrics[f"accuracy"] = accuracies["precision_at_1"]

        return metrics
