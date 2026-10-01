"""Order embeddings (Vendrov et al. 2016). Each skill is a point with non-negative coordinates, specific skills
further from the origin; A implies B when A ≥ B on every coordinate.
s(A→B) = −Σᵢ max(0, f(B)ᵢ − f(A)ᵢ)², the order violation, 0 iff A ⊑ B."""
import torch.nn.functional as F

from .base import Head, free_vectors, max_margin_loss, text_mlp


class OrderEmbedding(Head):
    def __init__(self, dim=64, margin=1.0, n_nodes=None):
        super().__init__()
        self.margin = margin
        self.encoder = free_vectors(n_nodes, dim) if n_nodes else text_mlp(dim)

    def embed(self, x):
        return self.encoder(x).abs()

    def dist(self, a, b):
        return F.relu(b - a).pow(2).sum(-1)

    def score(self, a, b):
        return -self.dist(a, b)

    def loss(self, score, y):
        return max_margin_loss(score, y, self.margin)
