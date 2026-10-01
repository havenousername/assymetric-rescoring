"""Tests for two claims that were guesses in docs/results.md. Task B test pairs, 3 seeds, sweep configs.
  1. Why is hyp_trans the best free-vector model? Candidate causes:
       curvature     → euc_trans: same model, loss and training, flat space
       N&K loss      → order_nk_trans: order geometry trained with hyp's softmax loss
       neighbourhood → vote: no learning; A inherits the ancestors of skills with similar graph neighbourhoods
       popularity    → pop: no learning; every list ranked by generality; vote+pop = vote with pop as backoff
  2. Why does text generalize? Hit rate split by whether B's label appears in A's encoder text.
Strata per hidden pair (A, B): nb = vote(A, B) > 0 (some graph neighbour of A has ancestor B);
named = B's label occurs in text(A), the string the encoder embedded.
Usage: uv run python -m experiments.probe"""
import json
import re
import sys

import numpy as np
import torch

from skillmatch.data import SkillGraph
from skillmatch.methods import cosine, cosine_generality
from skillmatch.methods.neighbours import popularity, vote, vote_matrix, vote_popularity
from skillmatch.tasks import evaluate, up_known

from .zoo import trained

VOTE_POP_LAMBDAS = (0.01, 0.03, 0.1, 0.3, 1)  # picked on val B MAP
MODELS = ["box_trans", "order_trans", "order_nk_trans", "transe_trans", "euc_trans", "hyp_trans", "box", "order", "euc", "hyp"]


class Probe:
    """Per-pair filtered hit@10 on the Task B test pairs, and the strata."""

    def __init__(self, graph):
        self.graph, self.V = graph, vote_matrix(graph)
        self.relevant = up_known(graph, "test")
        self.col = {c: j for j, c in enumerate(graph.catalogue)}
        self.pairs = [(a, b) for a in self.relevant for b in sorted(self.relevant[a]) if b in self.col]
        V = self.V
        self.strata = {
            "all": lambda p: True,
            "nb": lambda p: V[p] > 0, "no nb": lambda p: V[p] == 0,
            "named": lambda p: self.named(*p), "not named": lambda p: not self.named(*p),
            "not named, nb": lambda p: not self.named(*p) and V[p] > 0,
            "not named, no nb": lambda p: not self.named(*p) and V[p] == 0}

    def named(self, a, b):
        return re.search(rf"(?<!\w){re.escape(self.graph.label(b))}(?!\w)", self.graph.text(a), re.I) is not None

    def cells(self, names=None):
        return {k: [p for p in self.pairs if f(p)] for k, f in self.strata.items() if names is None or k in names}

    def hits10(self, M):
        """P(B in A's filtered top 10) per pair, ties broken at random: box (log P clamped at −1e-6), order (energy 0)
        and vote (counts) tie a lot, and counting ties as wins inflates them."""
        g, col = self.graph, self.col
        S, out = M(torch.tensor(list(self.relevant)), torch.tensor(g.catalogue)).detach().cpu(), {}
        for s, (a, rel) in zip(S, self.relevant.items()):
            s = s.clone()
            s[[col[e] for e in g.ancestors_train[a] | {a} if e in col]] = -float("inf")
            for b in rel & set(col):
                t = s.clone()
                t[[col[o] for o in rel - {b} if o in col]] = -float("inf")  # the other answers don't count
                above, tied = int((t > t[col[b]]).sum()), int((t == t[col[b]]).sum())
                out[(a, b)] = min(max((10 - above) / tied, 0.0), 1.0)
        return out


def vote_pop_lambda(graph, V):
    return max(VOTE_POP_LAMBDAS, key=lambda l: evaluate(graph, vote_popularity(V, graph, l), "val", node_task=False)["B"]["MAP"])


if __name__ == "__main__":
    graph = SkillGraph()
    probe = Probe(graph)
    cells = probe.cells()
    print(f"{len(probe.pairs)} Task B test pairs; strata sizes: " + ", ".join(f"{k} {len(v)}" for k, v in cells.items()))
    V = probe.V
    with torch.no_grad():
        lam = vote_pop_lambda(graph, V)
        baselines = {"vote": vote(V), "pop": popularity(graph), "vote+pop": vote_popularity(V, graph, lam)}
        for name, M in baselines.items():
            print(f"{name}, Task B test: {json.dumps(evaluate(graph, M, 'test', node_task=False)['B'], default=lambda x: round(float(x), 3))}")
        rows = {"cosine": [probe.hits10(cosine(graph.X))], "cos+gen": [probe.hits10(cosine_generality(graph.X, graph.generality))],
                **{name: [probe.hits10(M)] for name, M in baselines.items()}}
    for key in MODELS:
        for seed in (0, 1, 2):
            _, M = trained(graph, key, seed)
            with torch.no_grad(): rows.setdefault(key, []).append(probe.hits10(M))
        print(key, "done", file=sys.stderr)
    print(f"\nhit@10 per stratum (mean over seeds; the first five rows are deterministic)\n{'model':15}" + "".join(f"{k:>18}" for k in cells))
    for key, rs in rows.items():
        h = [np.mean([np.mean([r[p] for p in c]) for r in rs]) if c else float("nan") for c in cells.values()]
        print(f"{key:15}" + "".join(f"{x:18.3f}" for x in h))
