import torch

from pytorch_metric_learning.reducers import AvgNonZeroReducer
from pytorch_metric_learning.utils import loss_and_miner_utils as lmu
from pytorch_metric_learning.losses.generic_pair_loss import GenericPairLoss
from pytorch_metric_learning.distances import LpDistance


class SoftContrastiveLoss(GenericPairLoss):
    def __init__(self, pos_margin=1, neg_margin=0, **kwargs):
        super().__init__(mat_based_loss=False, **kwargs)
        self.pos_margin = pos_margin
        self.neg_margin = neg_margin
        self.add_to_recordable_attributes(
            list_of_names=["pos_margin", "neg_margin"], is_stat=False
        )


        self.A = torch.nn.Parameter(torch.Tensor([1]))
        # Constraint A to be positive
        self.A.data.clamp_(min=0)

        self.B = torch.nn.Parameter(torch.Tensor([1]))

    def _compute_loss(self, pos_pair_dist, neg_pair_dist, indices_tuple):
        pos_loss, neg_loss = 0, 0
        if len(pos_pair_dist) > 0:
            pos_loss = self.get_per_pair_loss(pos_pair_dist, "pos")
        if len(neg_pair_dist) > 0:
            neg_loss = self.get_per_pair_loss(neg_pair_dist, "neg")
        pos_pairs = lmu.pos_pairs_from_tuple(indices_tuple)
        neg_pairs = lmu.neg_pairs_from_tuple(indices_tuple)
        return {
            "pos_loss": {
                "losses": pos_loss,
                "indices": pos_pairs,
                "reduction_type": "pos_pair",
            },
            "neg_loss": {
                "losses": neg_loss,
                "indices": neg_pairs,
                "reduction_type": "neg_pair",
            },
        }

    def get_per_pair_loss(self, pair_dists, pos_or_neg):
        loss_calc_func = self.pos_calc if pos_or_neg == "pos" else self.neg_calc
        margin = self.pos_margin if pos_or_neg == "pos" else self.neg_margin
        per_pair_loss = loss_calc_func(pair_dists, margin)
        return per_pair_loss

    def pos_calc(self, pos_pair_dist, margin):
        distance = self.distance(pos_pair_dist, margin)
        p = torch.sigmoid(-self.A * distance + self.B)

        return -torch.log(p)

    def neg_calc(self, neg_pair_dist, margin):
        distance = self.distance(neg_pair_dist, margin)
        p = torch.sigmoid(-self.A * distance + self.B)

        return -torch.log(1 - p)

    def get_default_reducer(self):
        return AvgNonZeroReducer()

    def _sub_loss_names(self):
        return ["pos_loss", "neg_loss"]
    
    def get_default_distance(self):
        return LpDistance(p=2)