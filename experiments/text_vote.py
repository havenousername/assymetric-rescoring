"""The four transformer-free text graphs combined by vote, no embedding, no model: mention, genus, label
(experiments/local_graph.py) and invCL over co-mentions at m = 10 (experiments/dih.py). Each source votes A → B;
net = votes(A → B) − votes(B → A), p = net / 4. Rules:
- union:           net ≥ 1;
- vote ≥ 2:        net ≥ 2;
- strong ∪ weak²:  genus ∪ label (precision ~0.6) ∪ (mention ∩ invCL comention) (precision ~0.2 each).
Scored like local_graph.report (cos + gen, tasks A–D, identity, edge stats) with gen hard or soft (p) and direct or
closure picked on val A MAP. Usage: uv run python -m experiments.text_vote → results/text_vote.json"""
import sys
import time
from collections import Counter

import torch

from skillmatch.data import SkillGraph
from skillmatch.methods import cosine_generality

from .dih import comention_features, edges_topm, measures
from .knn_nli import collection, scored
from .local_graph import genus, label, line, mention, stats
from .nollm_gen import closure, generality, row
from .zoo import save


def sources(graph):
    coll = collection(graph)
    invcl = measures(comention_features(graph, coll))[2]
    S = {"mention": mention(graph), "genus": genus(graph), "label": label(graph), "invCL comention": edges_topm(graph, coll, invcl, 10)}
    return {k: {(a, b) for a, b, _ in E} for k, E in S.items()}


def vote(S, keep):
    """(a, b, p) for the pairs whose net vote passes keep(net, sources voting a → b)."""
    votes = Counter(e for E in S.values() for e in E)
    out = []
    for (a, b), v in votes.items():
        net = v - votes.get((b, a), 0)
        if net > 0 and keep(net, {k for k, E in S.items() if (a, b) in E}): out.append((a, b, net / len(S)))
    return out


RULES = {"union": lambda net, s: True,
         "vote ≥ 2": lambda net, s: net >= 2,
         "strong ∪ weak²": lambda net, s: bool(s & {"genus", "label"}) or {"mention", "invCL comention"} <= s}


def report(graph, E, seconds):
    grid = {(soft, d): row(graph, E, soft, d) for soft in (False, True) for d in (True, False)}
    soft, direct = c = max(grid, key=lambda c: grid[c]["val A MAP"])
    r = grid[c]
    gen = generality(graph, closure(graph, E, direct)[0], soft)
    return {**r, **scored(graph, cosine_generality(graph.X, gen, r["λ"])), "soft": soft, "direct": direct, "build s": seconds, **stats(graph, E)}


if __name__ == "__main__":
    graph, res = SkillGraph(), {}
    t = time.time()
    S = sources(graph)
    built = time.time() - t
    with torch.no_grad():
        for name, keep in RULES.items():
            E = vote(S, keep)
            assert not {(b, a) for a, b, _ in E} & {(a, b) for a, b, _ in E}, "both directions kept"
            res[name] = report(graph, E, built)
            print(line(name, res[name]), "| soft", res[name]["soft"], "direct", res[name]["direct"], file=sys.stderr)
    best = max(res, key=lambda k: res[k]["val A MAP"])
    res["best on val"] = best
    save("text_vote.json", res)
    print("best on val A:", best, file=sys.stderr)
