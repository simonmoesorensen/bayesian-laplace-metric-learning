import torch
from pytorch_metric_learning import distances, miners

class AllCombinationsMiner:
    """
    Returns all combinations of two points, except for points with themselves.
    """

    def __call__(self, _, y):
        batch_size = y.shape[0]
        indices = torch.arange(batch_size)
        pairs = torch.combinations(indices)
        mask = y[pairs[:, 0]] == y[pairs[:, 1]]
        a1, p = pairs[mask, 0], pairs[mask, 1]
        a2, n = pairs[~mask, 0], pairs[~mask, 1]
        return (a1, p, a2, n)


class AllPermutationsMiner:
    """
    Returns all permutations of two points, except for points with themselves.
    """

    def __init__(self) -> None:
        self._miner = miners.BatchEasyHardMiner(
            pos_strategy="all",
            neg_strategy="all",
            distance=distances.LpDistance(normalize_embeddings=False, p=2, power=1),
            allowed_pos_range=None,
            allowed_neg_range=None,
        )

    def __call__(self, x, y):
        return self._miner(x, y)


class AllPositiveMiner:
    """
    Returns all permutations of positive points.
    """

    def __init__(self) -> None:
        self._miner = miners.BatchEasyHardMiner(
            pos_strategy="all",
            neg_strategy="hard",
            distance=distances.LpDistance(normalize_embeddings=False, p=2, power=1),
            allowed_pos_range=None,
            allowed_neg_range=(torch.inf, torch.inf),
        )

    def __call__(self, x, y):
        return self._miner(x, y)

class AllNegativeMiner:
    """
    Returns all permutations of negative points.
    """

    def __init__(self) -> None:
        self._miner = miners.BatchEasyHardMiner(
            pos_strategy="hard",
            neg_strategy="all",
            distance=distances.LpDistance(normalize_embeddings=False, p=2, power=1),
            allowed_pos_range=(torch.inf, torch.inf),
            allowed_neg_range=None,
        )

    def __call__(self, x, y):
        return self._miner(x, y)
