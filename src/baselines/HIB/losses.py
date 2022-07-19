import torch

from pytorch_metric_learning.reducers import AvgNonZeroReducer
from pytorch_metric_learning.utils import loss_and_miner_utils as lmu
from pytorch_metric_learning.losses.generic_pair_loss import GenericPairLoss
from pytorch_metric_learning.distances import LpDistance
from pytorch_metric_learning.utils import common_functions as c_f
from pytorch_metric_learning.losses.mixins import WeightMixin


class WeightClipper:
    def __init__(self, frequency=5):
        self.frequency = frequency

    def __call__(self, module):
        # filter the variables to get the ones you want
        if hasattr(module, 'A'):
            module.A.data = module.A.data.clamp(min=0)

class SoftContrastiveLoss(WeightMixin, GenericPairLoss):
    def __init__(self, **kwargs):
        super().__init__(mat_based_loss=False, **kwargs)
        self.add_to_recordable_attributes(
            list_of_names=["A", "B"], is_stat=False
        )

        self.A = torch.nn.Parameter(torch.Tensor(1))
        # Constrain A to be positive
        self.weight_init_func(self.A)
        self.weight_clipper = WeightClipper()

        self.B = torch.nn.Parameter(torch.Tensor(1))
        self.weight_init_func(self.B)


    def cast_params(self, device):
        self.A.data = self.A.data.to(device)
        self.B.data = self.B.data.to(device)

    def _compute_loss(self, pos_pair_dist, neg_pair_dist, indices_tuple):
        self.cast_params(pos_pair_dist.device)

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
        p = torch.clamp(p, min=1e-8, max=1.0)

        return p

    def pos_calc(self, pos_pair_dist):
        p = self.p_calc(pos_pair_dist)

        return -torch.log(p)

    def neg_calc(self, neg_pair_dist):
        p = self.p_calc(neg_pair_dist)

        return -torch.log(1 - p)

    def get_default_reducer(self):
        return AvgNonZeroReducer()

    def _sub_loss_names(self):
        return ["pos_loss", "neg_loss"]
    
    def get_default_distance(self):
        return LpDistance(p=2)



# class KLpDistance(BaseDistance):
#     def __init__(self, K, **kwargs):
#         super().__init__(**kwargs)
#         assert not self.is_inverted
#         self.K = K

#     def compute_mat(self, query_emb, ref_emb):
#         dtype, device = query_emb.dtype, query_emb.device
#         if ref_emb is None:
#             ref_emb = query_emb

#         # Repeat over `self.K` samples
        
#         # Repeat interleave tensor so something like [[1,2],[3,4],[4,5]] becomes 
#         # [[1,2],[1,2],[3,4],[3,4],[4,5],[4,5]]
#         query_emb_K = query_emb.repeat_interleave(self.K**2, dim=0)

#         # Repeat tensor so something like [[1,2],[3,4],[4,5]] becomes
#         # [[1,2],[3,4],[4,5],[1,2],[3,4],[4,5]]
#         ref_emb_K = ref_emb.repeat(self.K**2, 1, 1)

#         out = []
#         for k in range(self.K**2):
#             query_emb_sample = query_emb_K[k]
#             ref_emb_sample = ref_emb_K[k]

#             # Compute pairwise distances
#             if dtype == torch.float16:  # cdist doesn't work for float16
#                 rows, cols = lmu.meshgrid_from_sizes(query_emb_sample, ref_emb_sample, dim=0)
#                 output = torch.zeros(rows.size(), dtype=dtype, device=device)
#                 rows, cols = rows.flatten(), cols.flatten()
#                 distances = self.pairwise_distance(query_emb_sample[rows], ref_emb_sample[cols])
#                 output[rows, cols] = distances
#                 out.append(output)
#             else:
#                 out.append(torch.cdist(query_emb_sample, ref_emb_sample, p=self.p))

#         return out

#     def pairwise_distance(self, query_emb, ref_emb):
#         return torch.nn.functional.pairwise_distance(query_emb, ref_emb, p=self.p)
