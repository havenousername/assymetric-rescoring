"""No candidate stage: the NLI judge scores every ordered pair of the collection (941 × 940), so no pair is ever
missed for not being an embedding neighbour (Chen, Lin & Klein 2021: scoring all pairs and picking parents per
node recovers what threshold gating on a candidate list loses). Graphs, all picked on val A MAP:
- t:        p(A→B) ≥ t and p(A→B) > p(B→A), as in knn_nli;
- top-m:    B among A's m highest p(A→B), same direction rule;
- cascade:  the small judge's top-m parents per node re-judged by the large NLI (hub_nli cache), kept at p ≥ t.
Scores are cached as a dense [n, n] matrix in results/all_pairs_<key>.npy (row A, column B).
Usage: uv run python -m experiments.all_pairs [model key ...] (local_judge.NLIS) → results/all_pairs.json"""
import itertools
import sys
import time

import numpy as np
import torch

from skillmatch.data import ROOT, SkillGraph

from .hub_nli import edges as nli_edges, judged
from .knn_nli import THRESHOLDS, collection, text
from .local_graph import line, report
from .local_judge import NLIS, nli_scorer
from .zoo import load, save

MS = (1, 2, 3, 5, 10)


def matrix(graph, key):
    path = ROOT / "results" / f"all_pairs_{key}.npy"
    if path.exists(): return np.load(path), 0.0
    coll = collection(graph)
    score, t = nli_scorer(NLIS[key], "person"), time.time()
    T = [text(graph, i) for i in coll]
    P = np.zeros((len(coll), len(coll)), dtype=np.float16)
    for r in range(len(coll)):
        P[r] = score([(T[r], T[c]) for c in range(len(coll))], batch=128)
        if r % 50 == 0: print(f"  {key} row {r}/{len(coll)} {(time.time() - t) / (r + 1):.1f} s/row", file=sys.stderr, flush=True)
    P[np.arange(len(coll)), np.arange(len(coll))] = 0
    np.save(path, P)
    return P, time.time() - t


def threshold_edges(graph, coll, P, t):
    A, B = np.nonzero((P >= t) & (P > P.T))
    return [(graph.ids[coll[a]], graph.ids[coll[b]], float(P[a, b])) for a, b in zip(A, B)]


def topm_edges(graph, coll, P, m, t=0.0):
    out = []
    for a in range(len(coll)):
        for b in np.argsort(-P[a])[:m]:
            if P[a, b] > P[b, a] and P[a, b] >= t: out.append((graph.ids[coll[a]], graph.ids[coll[b]], float(P[a, b])))
    return out


def best(rows):
    k = max(rows, key=lambda k: rows[k]["val A MAP"])
    return {**rows[k], "picked": k, "per config": {str(c): {"A": r["A"]["MAP"], "B": r["B"]["MAP"], "val A": r["val A MAP"], "edges": r["edges"],
                                                     "prec": r["precision vs closure"]} for c, r in rows.items()}}


if __name__ == "__main__":
    graph, res = SkillGraph(), load("all_pairs.json")
    coll = collection(graph)
    for key in sys.argv[1:] or ["nli-modernbert"]:
        P, sec = matrix(graph, key)
        P = P.astype(np.float32)
        with torch.no_grad():
            res[f"all pairs {key}, t"] = best({t: report(graph, threshold_edges(graph, coll, P, t), sec) for t in THRESHOLDS})
            print(line(f"all pairs {key}, t", res[f"all pairs {key}, t"]), "picked", res[f"all pairs {key}, t"]["picked"], file=sys.stderr)
            res[f"all pairs {key}, top-m"] = best({(m, t): report(graph, topm_edges(graph, coll, P, m, t), sec) for m in MS for t in (0.0, 0.3, 0.5)})
            print(line(f"all pairs {key}, top-m", res[f"all pairs {key}, top-m"]), "picked", res[f"all pairs {key}, top-m"]["picked"], file=sys.stderr)
            save("all_pairs.json", res)
            if key != "nli":
                rows = {}
                for m in (3, 5, 10):
                    pairs = {tuple(sorted((coll[a], coll[b]))) for a in range(len(coll)) for b in np.argsort(-P[a])[:m]}
                    S, nli_s, new = judged(graph, pairs)
                    for t in THRESHOLDS: rows[m, t] = {**report(graph, nli_edges(graph, S, t), sec + nli_s), "pairs": len(pairs)}
                res[f"cascade {key} → nli"] = best(rows)
                print(line(f"cascade {key} → nli", res[f"cascade {key} → nli"]), "picked", res[f"cascade {key} → nli"]["picked"], file=sys.stderr)
                save("all_pairs.json", res)
