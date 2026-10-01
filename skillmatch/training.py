"""The two training loops. Both see only the training graph: positives are its transitive closure (A, B), A → B.

train_pairwise (dual, box, order, TransE, pair MLP): each epoch, every positive plus three negatives,
  the reversed pair (B, A), a random catalogue skill A doesn't imply, and a cousin (another descendant of B that
  A doesn't imply). The head's own loss scores them.
train_softmax (hyperbolic, Nickel & Kiela 2017): a softmax over −dist that has to pick B out of B plus k skills
  unrelated to A in either direction.
inputs: the encoder input per node, graph.X for text heads, graph.node_ids for free vectors.
"""
import random

import torch
import torch.nn.functional as F

from .data import DEVICE


def negatives(graph, a, b):
    out = [(b, a)]
    while True:
        r = random.choice(graph.catalogue)
        if r != a and r not in graph.ancestors_train[a]:
            out.append((a, r))
            break
    cousins = [c for c in graph.descendants_train[b] if c != a and c not in graph.ancestors_train[a]]
    if cousins: out.append((a, random.choice(cousins)))
    return out


def labelled_pairs(graph):
    """One epoch of (A, B, y), shuffled: y = 1 for the positives, 0 for their negatives."""
    pairs = [(a, b, 1.0) for a, b in graph.positives] + [(u, v, 0.0) for a, b in graph.positives for u, v in negatives(graph, a, b)]
    random.shuffle(pairs)
    return pairs


def train_pairwise(model, graph, epochs, lr, inputs=None, batch=512):
    inputs = graph.X if inputs is None else inputs
    model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        pairs = labelled_pairs(graph)
        for i in range(0, len(pairs), batch):
            a, b, y = (torch.tensor(t, device=DEVICE) for t in zip(*pairs[i:i + batch]))
            loss = model.loss(model(inputs[a], inputs[b]), y.float())
            opt.zero_grad(); loss.backward(); opt.step()
    return model.eval()


def train_softmax(model, graph, epochs, lr, inputs=None, k=10, batch=64):
    """Needs model.embed and model.dist. ponytail: Adam on tangent vectors (expmap0) instead of N&K's Riemannian
    SGD with burn-in; switch if the fit stalls."""
    inputs = graph.X if inputs is None else inputs
    related = {a: graph.ancestors_train[a] | set(graph.descendants_train[a]) | {a} for a in graph.catalogue}

    def unrelated(a):
        out = []
        while len(out) < k:
            r = random.choice(graph.catalogue)
            if r not in related[a]: out.append(r)
        return out

    model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        shuffled = random.sample(graph.positives, len(graph.positives))
        for i in range(0, len(shuffled), batch):
            a, b = zip(*shuffled[i:i + batch])
            candidates = torch.tensor([[pb] + unrelated(pa) for pa, pb in zip(a, b)], device=DEVICE)  # true B first
            d = model.dist(model.embed(inputs[torch.tensor(a, device=DEVICE)])[:, None], model.embed(inputs[candidates]))
            loss = F.cross_entropy(-d, torch.zeros(len(a), dtype=torch.long, device=DEVICE))
            opt.zero_grad(); loss.backward(); opt.step()
    return model.eval()
