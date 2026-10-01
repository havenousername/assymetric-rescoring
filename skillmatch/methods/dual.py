"""Dual encoder: two heads on the same text vector, one per role. s(A→B) = src(A)·tgt(B).
Asymmetric because src ≠ tgt. Serving: store tgt(B), search with src(A); Qdrant Dot distance, native HNSW."""
from .base import Head, text_mlp


class DualEncoder(Head):
    def __init__(self, dim=64):
        super().__init__()
        self.src, self.tgt = text_mlp(dim), text_mlp(dim)

    def forward(self, xa, xb):
        return (self.src(xa) * self.tgt(xb)).sum(-1)

    def matrix(self, xa, xb):
        return self.src(xa) @ self.tgt(xb).T
