"""Combinations on top of experiments/hub_nli.py candidate sets, no new candidates:
- agree:   an NLI edge (p ≥ t, NLI direction) is kept only when the LM judge (experiments/lm_prob.py, cached scores)
           prefers the same direction; t on val A MAP.
- +mention gen: the NLI graph's gen plus the graph-free mention count, gen = log(1 + #descendants) + log(1 + #mentions),
           one λ on val (the query stays cos + λ·gen).
Usage: uv run python -m experiments.ensemble <set ...> → results/ensemble.json"""
import sys
import time

import torch

from skillmatch.data import SkillGraph
from skillmatch.methods import cosine_generality

from .hub_nli import SETS, edges, judged
from .knn_nli import THRESHOLDS, scored
from .local_graph import line, mention_gen, report, stats
from .nollm_gen import closure, cos_gen, generality
from .zoo import load, save


def lm_scores(P):
    C = next(iter(load("lm_prob_scores.json").values()), {})
    return {p: tuple(C[f"{p[0]},{p[1]}"]) for p in P if f"{p[0]},{p[1]}" in C}


def agree(graph, S, L, t):
    """NLI edges whose direction the LM judge shares (pairs the LM has not scored are kept)."""
    keep = []
    for a, b, p in edges(graph, S, t):
        i, j = graph.index[a], graph.index[b]
        f, r = L.get((min(i, j), max(i, j)), (0, 0))
        if i > j: f, r = r, f
        if f >= r: keep.append((a, b, p))
    return keep


def with_mention_gen(graph, E, mg):
    r0 = report(graph, E, 0.0)
    gen = generality(graph, closure(graph, E, r0["direct"])[0]) + mg
    r = cos_gen(graph, gen)
    return {**r, **scored(graph, cosine_generality(graph.X, gen, r["λ"])), **stats(graph, E), "direct": r0["direct"]}


if __name__ == "__main__":
    graph, res = SkillGraph(), load("ensemble.json")
    mg = mention_gen(graph)
    with torch.no_grad():
        for name in sys.argv[1:]:
            t0 = time.time()
            P = SETS[name](graph)
            S, _, _ = judged(graph, P)
            L = lm_scores(P)
            if len(L) == len(P):
                rows = {t: report(graph, agree(graph, S, L, t), time.time() - t0) for t in THRESHOLDS}
                best = max(rows, key=lambda t: rows[t]["val A MAP"])
                res[f"agree {name}"] = {**rows[best], "t": best}
                print(line(f"agree {name}", res[f"agree {name}"]), "t", best, file=sys.stderr)
            else: print(f"{name}: LM scores cover {len(L)}/{len(P)} pairs, agreement skipped", file=sys.stderr)
            t = load("hub_nli.json")[name]["t"]
            res[f"{name} +mention gen"] = with_mention_gen(graph, edges(graph, S, t), mg)
            print(line(f"{name} +mention gen", res[f"{name} +mention gen"]), file=sys.stderr)
            save("ensemble.json", res)
