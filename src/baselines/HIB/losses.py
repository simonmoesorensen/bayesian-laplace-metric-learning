import torch
from pytorch_metric_learning import reducers
from pytorch_metric_learning.distances import LpDistance
from pytorch_metric_learning.losses.generic_pair_loss import GenericPairLoss
from pytorch_metric_learning.losses.mixins import WeightMixin
from pytorch_metric_learning.utils import loss_and_miner_utils as lmu


class WeightClipper:
    def __init__(self, frequency=5):
        self.frequency = frequency

    def __call__(self, module):
        # filter the variables to get the ones you want
        if hasattr(module, "A"):
            module.A.data = module.A.data.clamp(min=1e-6)


class SoftContrastiveLoss(WeightMixin, GenericPairLoss):
    def __init__(self, **kwargs):
        super().__init__(mat_based_loss=False, **kwargs)
        self.add_to_recordable_attributes(list_of_names=["A", "B"], is_stat=False)
        self.weight_clipper = WeightClipper()

        self.A = torch.nn.Parameter(torch.Tensor(1))
        # Constrain A to be positive
        self.weight_init_func(self.A)

        self.B = torch.nn.Parameter(torch.Tensor(1))
        self.weight_init_func(self.B)

    def cast_params(self, device):
        self.A.data = self.A.data.to(device)
        self.B.data = self.B.data.to(device)

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
        per_pair_loss = loss_calc_func(pair_dists)
        return per_pair_loss

    def p_calc(self, pair_dist):
        p = torch.sigmoid(-self.A * pair_dist + self.B)

        # Numerical stability, clamp to 1e-14
        p = p.clamp(min=1e-14)
        return p

    def pos_calc(self, pos_pair_dist):
        p = self.p_calc(pos_pair_dist)

        return -torch.log(p)

    def neg_calc(self, neg_pair_dist):
        p = self.p_calc(neg_pair_dist)

        return -torch.log(1 - p)

    def get_default_reducer(self):
        return reducers.MeanReducer()

    def _sub_loss_names(self):
        return ["pos_loss", "neg_loss"]

    def get_default_distance(self):
        return LpDistance(p=2, normalize_embeddings=False)
