import torch

from pytorch_metric_learning import reducers

from pytorch_metric_learning.losses import BaseMetricLossFunction


class MLSLoss(BaseMetricLossFunction):
    """
    Minimum likelihood score loss.
    """

    def compute_loss(self, embeddings, labels, indices_tuple, ref_emb, ref_labels):
        # Only get the positive pairs
        if len(indices_tuple) == 3:
            anc, pos, _ = indices_tuple
        elif len(indices_tuple) == 4:
            anc, pos, _, _ = indices_tuple

        mu, sigma_sq = embeddings, ref_emb

        # [batch_size, batch_size]
        loss_mls = self.negative_MLS(mu, mu, sigma_sq, sigma_sq)

        # Only get the loss for 'genuine' aka positive pairs
        loss = loss_mls[anc, pos]

        return {
            "loss": {
                "losses": loss,
                "indices": (anc, pos),
                "reduction_type": "pos_pair",
            }
        }

    def negative_MLS(self, X, Y, sigma_sq_X, sigma_sq_Y):
        embedding_size = X.shape[1]

        # Reshape to matrix of [batch_size, embedding_size, embedding_size]
        X = X.reshape([-1, 1, embedding_size])
        Y = Y.reshape([1, -1, embedding_size])
        sigma_sq_X = sigma_sq_X.reshape([-1, 1, embedding_size])
        sigma_sq_Y = sigma_sq_X.reshape([1, -1, embedding_size])

        sigma_sq_fuse = sigma_sq_X + sigma_sq_Y

        # From equation (3) in the paper
        diffs = (
            ((X - Y).square() / (1e-10 + sigma_sq_fuse)) + sigma_sq_fuse.log()
        )

        return diffs.sum(axis=2)

    def get_default_reducer(self):
        return reducers.MeanReducer()
