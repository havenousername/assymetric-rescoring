"""Poincaré embeddings (Nickel & Kiela 2017), the geometry of the Qdrant hyperbolic article (Mahmood & Kupchanko 2026).
Skills are points in the unit ball; general skills settle near the origin. The distance is symmetric, so the
article's score s(A→B) = −d(A, B) has no direction. alpha > 0 gives N&K's is-a score
s(A→B) = −(1 + α(‖B‖ − ‖A‖))·d(A, B), which also asks B to sit closer to the origin than A.
mu > 0 adds a depth term for the down direction: s(A→B) = ... + μ·d(0, A). For a fixed A (up: what does A imply?)
it is a constant; for a fixed B (down: who has B?) it favours profiles deeper in the ball, i.e. more specific than B.
At μ = 1 the score is d(0, A) − d(A, B), which peaks when B lies on the path from the centre to A, as an ancestor does
in a tree. Trained with train_softmax (the N&K loss); alpha and mu only change the score, so they can be set after training."""
import torch

from .base import Head, free_vectors, text_mlp


def expmap0(v):
    """Tangent space at the origin → Poincaré ball (curvature −1)."""
    n = v.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    return torch.tanh(n) * v / n


def poincare_distance(u, v):
    uu, vv = (u * u).sum(-1).clamp(max=1 - 1e-5), (v * v).sum(-1).clamp(max=1 - 1e-5)
    x = (1 + 2 * ((u - v) ** 2).sum(-1) / ((1 - uu) * (1 - vv))).clamp_min(1 + 1e-7)
    return torch.log(x + torch.sqrt(x * x - 1))  # acosh, written as in the article's Formula Query


class PoincareEmbedding(Head):
    def __init__(self, dim=5, alpha=0.0, mu=0.0, n_nodes=None):
        super().__init__()
        self.alpha, self.mu = alpha, mu
        self.encoder = free_vectors(n_nodes, dim, init=1e-3) if n_nodes else text_mlp(dim)  # N&K init near the origin

    def embed(self, x):
        return expmap0(self.encoder(x))

    def dist(self, a, b):
        return poincare_distance(a, b)

    def score(self, a, b):
        return -(1 + self.alpha * (b.norm(dim=-1) - a.norm(dim=-1))) * self.dist(a, b) + self.mu * self.dist(torch.zeros_like(a), a)


class EuclideanEmbedding(PoincareEmbedding):
    """Ablation: the same model and training in flat space, to isolate what the curvature adds."""

    def embed(self, x):
        return self.encoder(x)

    def dist(self, a, b):
        return ((a - b) ** 2).sum(-1).clamp_min(1e-12).sqrt()
