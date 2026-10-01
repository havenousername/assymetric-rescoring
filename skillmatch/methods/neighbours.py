"""No-learning baselines on the training graph alone. They can't score a skill without edges.
pop:      s(A→B) = generality(B), the same list for everyone.
vote:     B's score is the Jaccard similarity-weighted count of A's graph neighbours (ancestors ∪ descendants)
          that have B as an ancestor.
vote+pop: vote with λ·pop as the backoff."""
import torch

from ..data import DEVICE


def vote_matrix(graph):
    """V[a, b] = Σ_w J(a, w)·[w → b], J = Jaccard over training neighbours. On the CPU."""
    n = len(graph.ids)
    R = torch.zeros(n, n)
    for w, bs in graph.ancestors_train.items(): R[w, list(bs)] = 1
    Nb = ((R + R.T) > 0).float()
    inter = Nb @ Nb.T
    J = (inter / (Nb.sum(1)[:, None] + Nb.sum(1)[None] - inter).clamp_min(1)).fill_diagonal_(0)
    return J @ R


def vote(V):
    return lambda a, b: V[a][:, b]


def popularity(graph):
    return lambda a, b: graph.generality[b.to(DEVICE)].expand(len(a), -1)


def vote_popularity(V, graph, lam):
    V = V.to(DEVICE)
    return lambda a, b: V[a.to(DEVICE)][:, b.to(DEVICE)] + lam * graph.generality[b.to(DEVICE)]
