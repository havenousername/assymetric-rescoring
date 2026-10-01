"""Recall at prefetch depth, generalization and training cost per method. Test split, 3 seeds, configs from the sweeps.
R@K = share of a query's true answers in its top K of the 925 catalogue skills; K = 10 / 50 / 100 / 200 ≈ 1 / 5 / 11 / 22%.
Serving is prefetch top K + rerank, so R@K at the prefetch depth caps anything a reranker can reach.
no-evidence hit@10: Task B pairs where no graph neighbour of the query has the answer (probe.py), i.e. not graph-derivable.
train s: wall time on this machine (M5 Pro, MPS), including (λ, μ) tuning for +gen+cos rows.
Usage: uv run python -m experiments.recall → results/recall.json"""
import sys
import time

import numpy as np
import torch

from skillmatch.data import DEVICE, SkillGraph
from skillmatch.methods import cosine, cosine_generality
from skillmatch.methods.neighbours import popularity, vote, vote_popularity
from skillmatch.tasks import evaluate

from .probe import Probe, vote_pop_lambda
from .zoo import folded, mean_flat, plus_hybrid, save, seed_all, trained

COLUMNS = ["A MAP", "A R@10", "A R@50", "A R@100", "A R@200", "B MAP", "B R@10", "B R@50", "B R@100", "B R@200",
           "no nb", "not named, no nb", "train s"]


def flat(r, hits, cells, secs):
    out = {f"{t} {k}": float(v) for t in ("A", "B") if t in r for k, v in r[t].items() if k != "n"}
    return {**out, **{k: float(np.mean([hits[p] for p in c])) for k, c in cells.items()}, "train s": secs}


def run(graph, probe, res, make, seeds=(0, 1, 2), node_task=True):
    """make() → {row: scorer}, trained once per seed; the wall time counts toward each of its rows."""
    cells, rows = probe.cells(("no nb", "not named, no nb")), {}
    for seed in seeds:
        seed_all(seed)
        t = time.time()
        scorers = make(seed)
        if DEVICE == "mps": torch.mps.synchronize()
        secs = time.time() - t
        with torch.no_grad():
            for name, M in scorers.items():
                rows.setdefault(name, []).append(flat(evaluate(graph, M, "test", node_task), probe.hits10(M), cells, secs))
    for name, rs in rows.items():
        res[name] = mean_flat(rs)
        print(name, "done", file=sys.stderr)
    save("recall.json", res)


if __name__ == "__main__":
    graph = SkillGraph()
    probe, res = Probe(graph), {}
    V = probe.V
    with torch.no_grad(): lam = vote_pop_lambda(graph, V)
    run(graph, probe, res, lambda s: {"cosine": cosine(graph.X), "pop": popularity(graph),
                                     "cos+gen": cosine_generality(graph.X, graph.generality)}, seeds=(0,))
    run(graph, probe, res, lambda s: {"vote": vote(V), "vote+pop": vote_popularity(V, graph, lam)}, seeds=(0,), node_task=False)
    run(graph, probe, res, lambda s: {"hyp (free vectors)": trained(graph, "hyp_trans", s)[1]}, node_task=False)
    run(graph, probe, res, lambda s: {"dual": trained(graph, "dual", s)[1]})
    for key in ("box", "order", "hyp", "transe"):
        run(graph, probe, res, lambda s, key=key: plus_hybrid(graph, key, trained(graph, key, s)[1]))
    run(graph, probe, res, lambda s: {"hyp+isa": trained(graph, "hyp+isa", s)[1]})
    run(graph, probe, res, lambda s: {"pair_mlp": trained(graph, "pair_mlp", s)[1]})
    run(graph, probe, res, lambda s: {"distilled dual, folded": folded(graph)[1]})
    print(f"{'model':24}" + "".join(f"{c:>9}" for c in ["A MAP", "R@10", "R@50", "R@100", "R@200", "B MAP", "R@10", "R@50",
                                                        "R@100", "R@200", "no-evid", "no-ev,unn", "train s"]))
    for name, r in res.items(): print(f"{name:24}" + "".join(f"{r[c]:9.3f}" if c in r else f"{'–':>9}" for c in COLUMNS))
