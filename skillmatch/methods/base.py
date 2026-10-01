"""What the trained heads share. A head sits on top of the frozen text encoder and scores s(A→B), how strongly
having skill A implies having skill B. Nothing here knows about the skill graph: heads take vectors in and give
scores out."""
import torch
import torch.nn.functional as F
from torch import nn

TEXT_DIM = 768  # bge-base


def text_mlp(out_dim, in_dim=TEXT_DIM, hidden=512):
    """The trainable part on top of the frozen encoder: one hidden layer."""
    return nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.Linear(hidden, out_dim))


def free_vectors(n_nodes, dim, init=None):
    """Ablation: one trainable vector per node and no text. It can't place a skill it never saw in training."""
    table = nn.Embedding(n_nodes, dim)
    if init: nn.init.uniform_(table.weight, -init, init)
    return table


def max_margin_loss(score, y, margin):
    """Vendrov et al.: the energy (−score) on positives, max(0, margin − energy) on negatives."""
    energy = -score
    return (y * energy + (1 - y) * F.relu(margin - energy)).mean()


class Head(nn.Module):
    """A trained scorer s(A→B). Subclasses define embed (encoder input → representation) and score (two
    representations → score, broadcasting over leading dims), or override forward and matrix."""

    def forward(self, xa, xb):
        """Aligned pairs: s(xa[i] → xb[i])."""
        return self.score(self.embed(xa), self.embed(xb))

    def matrix(self, xa, xb, chunk=64):
        """All pairs: [len(xa), len(xb)]."""
        ea, eb = self.embed(xa), self.embed(xb)
        return torch.cat([self.score(ea[i:i + chunk, None], eb[None]) for i in range(0, len(ea), chunk)])

    def loss(self, score, y):
        return F.binary_cross_entropy_with_logits(score, y)

    def scorer(self, inputs):
        """(A ids, B ids) → score matrix, the form the evaluation code takes. inputs: the encoder input per node,
        text vectors (graph.X), or node ids (graph.node_ids) for free vectors."""
        return lambda a, b: self.matrix(inputs[a], inputs[b])
