import torch


class AllPermutationsMiner:
    def __call__(self, y):
        batch_size = y.shape[0]
        indices = torch.arange(batch_size)
        pairs = torch.combinations(indices)
        mask = y[pairs[:, 0]] == y[pairs[:, 1]]
        a1, p = pairs[mask, 0], pairs[mask, 1]
        a2, n = pairs[~mask, 0], pairs[~mask, 1]
        return (a1, p, a2, n)
