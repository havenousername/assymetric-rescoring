"""Pair classifier: MLP([x_A, x_B, x_A ⊙ x_B, x_A − x_B]) → logit of P(B | A). Asymmetric because the input is
ordered. It has no per-skill vector, so it can only rerank a shortlist (or expand profiles at ingest)."""
import torch
from torch import nn

from .base import TEXT_DIM, Head


class PairMLP(Head):
    def __init__(self, hidden=512, in_dim=TEXT_DIM):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(4 * in_dim, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def forward(self, xa, xb):
        return self.net(torch.cat([xa, xb, xa * xb, xa - xb], -1)).squeeze(-1)

    def matrix(self, xa, xb, chunk=16):
        return torch.cat([self(xa[i:i + chunk, None].expand(-1, len(xb), -1), xb[None].expand(len(xa[i:i + chunk]), -1, -1))
                          for i in range(0, len(xa), chunk)])
