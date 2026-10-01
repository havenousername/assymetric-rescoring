"""Gumbel box embeddings (Dasgupta et al. 2020). Each skill is an axis-aligned box with soft edges;
A implies B when A's box sits inside B's. s(A→B) = log P(B | A) = log vol(A ∩ B) − log vol(A), at most 0.
A representation is [lo | hi], the two corners."""
import torch
import torch.nn.functional as F

from .base import Head, free_vectors, text_mlp

EULER_GAMMA = 0.5772156649  # Gumbel box volume correction


class BoxEmbedding(Head):
    def __init__(self, dim=64, beta_i=0.1, beta_v=1.0, n_nodes=None):
        """beta_i, beta_v: Gumbel temperatures of the intersection and the volume. n_nodes: free vectors (ablation)."""
        super().__init__()
        self.beta_i, self.beta_v = beta_i, beta_v
        self.encoder = free_vectors(n_nodes, 2 * dim) if n_nodes else text_mlp(2 * dim)  # centre, half-width

    def embed(self, x):
        centre, half = self.encoder(x).chunk(2, -1)
        half = F.softplus(half)
        return torch.cat([centre - half, centre + half], -1)

    def log_volume(self, lo, hi):
        return torch.log(F.softplus(hi - lo - 2 * EULER_GAMMA * self.beta_v, beta=1 / self.beta_v) + 1e-10).sum(-1)

    def score(self, a, b):
        (lo_a, hi_a), (lo_b, hi_b) = a.chunk(2, -1), b.chunk(2, -1)
        lo = self.beta_i * torch.logaddexp(lo_a / self.beta_i, lo_b / self.beta_i)  # soft max of the lower corners
        hi = -self.beta_i * torch.logaddexp(-hi_a / self.beta_i, -hi_b / self.beta_i)  # soft min of the upper corners
        return (self.log_volume(lo, hi) - self.log_volume(lo_a, hi_a)).clamp(max=-1e-6)

    def loss(self, score, y):
        """score = log P: −log P on positives, −log(1 − P) on negatives."""
        return -(y * score + (1 - y) * torch.log(-torch.expm1(score) + 1e-10)).mean()
