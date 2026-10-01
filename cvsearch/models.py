"""The scorers, built on a KeywordGraph. Only the model code comes from skillmatch (heads, trainers, hybrid).

Every scorer is s(A → B) over keyword ids, "having keyword A implies having keyword B", as M(a ids, b ids) → [len(a),
len(b)]. Keywords are read by name only (bge-base on "LLVM", "Spring Boot"): CVs carry no descriptions.
- cosine:                        cos(A, B), no training, symmetric.
- graph:                         the graph itself, no training: 2 if A = B, 1 + P(A → B) if the graph implies it (best
                                 path product of P(B | A)), 0 otherwise; 0.01·cos(A, B) orders what the graph leaves open.
- cos + gen:                     cos(A, B) + λ·gen(B), gen = log(1 + #descendants in the graph).
- order / box + gen + cos:       head(A, B) + λ·gen(B) + μ·cos(A, B), the head trained on the graph's closure.
- hyperbolic + gen + cos + depth: the same with a Poincaré head, plus μ_d·d(0, A) for the down direction.
Head configs are the methods' own defaults (picked on validation in the skill-graph study); the weights λ, μ, μ_d are
picked on CV val here (cvsearch.search.tune).
"""
import math
import random

import networkx as nx
import scipy.sparse as sp
import torch

from skillmatch.data import encode
from skillmatch.methods import BoxEmbedding, OrderEmbedding, PoincareEmbedding, cosine, cosine_generality, hybrid
from skillmatch.training import train_pairwise, train_softmax

HEADS = {  # name: (head, kwargs, trainer, lr)
    "order": (OrderEmbedding, {"dim": 128, "margin": 0.1}, train_pairwise, 1e-3),
    "box": (BoxEmbedding, {"dim": 128, "beta_i": 0.01}, train_pairwise, 1e-3),
    "hyp": (PoincareEmbedding, {"dim": 10}, train_softmax, 3e-3),
}
NAMES = {"cos": "cosine", "graph": "graph lookup", "cosgen": "cos + gen", "order": "order + gen + cos", "box": "box + gen + cos",
         "hyp": "hyperbolic + gen + cos + depth"}


def embed_keywords(vocab):
    """bge-base vectors of the keyword names, [V, 768] unit length."""
    return encode(list(vocab.surface))


def train(graph, head, epochs, seed=0):
    torch.manual_seed(seed)
    random.seed(seed)
    make, kw, trainer, lr = HEADS[head]
    return trainer(make(**kw), graph, epochs, lr)


def graph_lookup(graph):
    """The graph as a scorer: see the module docstring."""
    rows, cols, vals = [], [], []
    for a in graph.G:
        lengths = nx.single_source_dijkstra_path_length(graph.G, a, weight=lambda u, v, e: -math.log(e["weight"]))
        for b, d in lengths.items():
            rows.append(a); cols.append(b); vals.append(2.0 if a == b else 1 + math.exp(-d))
    C = sp.csr_matrix((vals, (rows, cols)), shape=(len(graph.vocab),) * 2)

    def M(a, b):
        a, b = a.cpu().numpy(), b.cpu().numpy()
        return torch.from_numpy(C[a][:, b].toarray()).float().to(graph.X.device) + 0.01 * (graph.X[a] @ graph.X[b].T)
    return M


def scorers(graph, heads, w):
    """{key: M} for every method. heads: {"order"|"box"|"hyp": trained model}; w: {key: weights} as tune() picks
    them, (λ,) for cos + gen, (λ, μ) for order and box, (λ, μ, μ_d) for hyp."""
    out = {"cos": cosine(graph.X), "graph": graph_lookup(graph),
           "cosgen": cosine_generality(graph.X, graph.generality, *w["cosgen"])}
    for k in ("order", "box", "hyp"):
        if k == "hyp": heads[k].mu = w[k][2]
        out[k] = hybrid(heads[k].scorer(graph.X), graph, *w[k][:2])
    return out
