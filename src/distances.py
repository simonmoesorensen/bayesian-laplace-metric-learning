import torch
from pytorch_metric_learning.distances import BaseDistance, LpDistance
from pytorch_metric_learning.utils import loss_and_miner_utils as lmu


class ExpectedSquareL2Distance(BaseDistance):
    def __init__(self, sample_dim=0, feature_dim=-1, **kwargs):
        super().__init__(**kwargs)

        self.l2_square_dist = LpDistance(p=2, power=2)
        self.sample_dim = sample_dim
        self.feature_dim = feature_dim

    def compute_mat(self, query_emb, ref_emb):
        assert len(query_emb.shape) == 3
        assert len(ref_emb.shape) == 3

        query_mean = query_emb.mean(dim=self.sample_dim)
        query_var = query_emb.var(dim=self.sample_dim)

        ref_mean = ref_emb.mean(dim=self.sample_dim)
        ref_var = ref_emb.var(dim=self.sample_dim)

        return (
            self.l2_square_dist.compute_mat(query_mean, ref_mean)
            + query_var.sum(dim=self.feature_dim).unsqueeze(1)
            + ref_var.sum(dim=self.feature_dim).unsqueeze(0)
        )

    def pairwise_distance(self, query_emb, ref_emb):
        assert len(query_emb.shape) == 3
        assert len(ref_emb.shape) == 3

        query_mean = query_emb.mean(dim=self.sample_dim)
        query_var = query_emb.var(dim=self.sample_dim)

        ref_mean = ref_emb.mean(dim=self.sample_dim)
        ref_var = ref_emb.var(dim=self.sample_dim)

        return (
            self.l2_square_dist.pairwise_distance(query_mean, ref_mean)
            + query_var.sum(dim=self.feature_dim)
            + ref_var.sum(dim=self.feature_dim)
        )
