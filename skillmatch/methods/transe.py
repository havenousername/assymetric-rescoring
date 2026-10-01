"""TransE (Bordes et al. 2013) with one relation r, "implies": s(A→B) = −‖f(A) + r − f(B)‖_p.
Serving: store f(B), search with f(A) + r; Qdrant Euclid (p = 2) or Manhattan (p = 1) distance, native HNSW."""
import torch
from torch import nn

from .base import Head, free_vectors, max_margin_loss, text_mlp


class TransE(Head):
    def __init__(self, dim=64, margin=1.0, p=2, n_nodes=None):
        super().__init__()
        self.margin, self.p = margin, p
        self.encoder = free_vectors(n_nodes, dim) if n_nodes else text_mlp(dim)
        self.r = nn.Parameter(0.1 * torch.randn(dim))

    def embed(self, x):
        return self.encoder(x)

    def dist(self, a, b):
        return (a + self.r - b).norm(p=self.p, dim=-1)

    def score(self, a, b):
        return -self.dist(a, b)

    def loss(self, score, y):
        return max_margin_loss(score, y, self.margin)
