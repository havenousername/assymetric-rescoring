"""generate_graph() for any collection: no LLM, no Wikidata, no Stack Overflow. Candidates = each skill's top-k
cosine neighbours in the collection (the skill graph minus its held-out skills); a local NLI judge
(experiments/local_judge.py, "person" template) scores both directions, and A → B is kept when p(A→B) ≥ t and
p(A→B) > p(B→A). Evaluation truth, the catalogue and val (which picks k, t, the gen variant, λ and μ) stay Claude's:
- cos + gen, gen counted hard or soft, on the closure or on the direct edges (experiments/nollm_gen.py), tasks A, B;
- order and order + gen + cos trained on the NLI graph instead of Claude's, 3 seeds, compare_sweep config; positives
  = its closure, or its direct edges only (negatives still avoid the closure);
- the down direction (Tasks C and D, skillmatch/tasks.py) for all three, direct and expanded, and identity (b's own profile
  first for b, skillmatch.tasks.identity).
Usage: uv run python -m experiments.knn_nli [model key ...] (local_judge.NLIS) → results/knn_nli.json
NLI scores are cached in results/knn_nli_scores.json."""
import copy
import itertools
import sys
import time

import numpy as np
import torch

from skillmatch.data import SkillGraph
from skillmatch.methods import cosine_generality, hybrid
from skillmatch.methods.hybrid import GRID
from skillmatch.tasks import down, down_known, evaluate, expanded, identity, val_score

from .local_judge import NLIS, nli_scorer
from .nollm_gen import closure, generality, row
from .zoo import load, mean_runs, save, trained

K_MAX, KS = 20, (5, 10, 20)
THRESHOLDS = (0.1, 0.3, 0.5, 0.7, 0.9)
SEEDS = (0, 1, 2)


def text(graph, i):
    """The "label — description" text the NLI templates read (local_judge.split)."""
    n = graph.nodes[graph.ids[i]]
    return f"{n['label']} — {n['desc'] or ''}"


def collection(graph):
    held = {i for s in graph.heldout.values() for i in s}
    return [i for i in range(len(graph.ids)) if i not in held]


def neighbours(graph):
    """[(i, j, rank)], i < j: collection pairs where one is in the other's top K_MAX by cosine, rank = the better of
    the two ranks (1 = nearest)."""
    coll = collection(graph)
    S = graph.X[coll] @ graph.X[coll].T
    S.fill_diagonal_(-2)
    best = {}
    for r, top in enumerate(S.topk(K_MAX, dim=1).indices.tolist()):
        for rank, c in enumerate(top, 1):
            pair = tuple(sorted((coll[r], coll[c])))
            best[pair] = min(best.get(pair, rank), rank)
    return [[i, j, r] for (i, j), r in sorted(best.items())]


def scores(graph, key, P):
    """{"pairs", "fwd": p(i→j), "rev": p(j→i), "s"} for the model NLIS[key], cached."""
    cache = load("knn_nli_scores.json")
    if cache.get(key, {}).get("pairs") != P:
        score, t = nli_scorer(NLIS[key], "person"), time.time()
        xy = [(text(graph, i), text(graph, j)) for i, j, _ in P]
        fwd, rev = score(xy), score([(y, x) for x, y in xy])
        cache[key] = {"pairs": P, "fwd": fwd, "rev": rev, "s": time.time() - t}
        save("knn_nli_scores.json", cache)
    return cache[key]


def edges(graph, s, k, t):
    """(specific, general, p) qid triples: the stronger direction of each pair within rank k, kept if p ≥ t."""
    out = []
    for (i, j, r), f, b in zip(s["pairs"], s["fwd"], s["rev"]):
        a, c, p = (i, j, f) if f > b else (j, i, b)
        if r <= k and p >= t: out.append((graph.ids[a], graph.ids[c], p))
    return out


def nli_graph(graph, pairs, gen, positives=None):
    """A copy of the skill graph whose training side (catalogue, positives, ancestors, descendants, generality) is the
    NLI graph's; evaluation (skillmatch/tasks.py) keeps reading Claude's truth from the original."""
    g = copy.copy(graph)
    g.catalogue = collection(graph)
    g.ancestors_train = {a: set() for a in g.catalogue}
    for a, b in pairs: g.ancestors_train[graph.index[a]].add(graph.index[b])
    g.descendants_train = {b: [a for a in g.ancestors_train if b in g.ancestors_train[a]] for b in range(len(graph.ids))}
    g.positives = [(a, b) for a, bs in g.ancestors_train.items() for b in bs] if positives is None else \
        [(graph.index[a], graph.index[b]) for a, b in positives]
    g.generality = gen
    return g


def scored(graph, S):
    """Tasks A, B, C and D (C and D direct and expanded) on Claude's truth, plus identity."""
    E = expanded(graph, S)
    return {**evaluate(graph, S, "test"), "C": down_known(graph, S), "C expanded": down_known(graph, E),
            "D": down(graph, S), "D expanded": down(graph, E),
            "identity": {k: float(v) for k, v in identity(graph, S).items()}}


def order_runs(graph, g, tag=""):
    """order and order + gen + cos trained on g, (λ, μ) on Claude's val, mean over seeds."""
    runs = {}
    for seed in SEEDS:
        _, M = trained(g, "order", seed)
        with torch.no_grad():
            w = max(GRID, key=lambda w: val_score(graph, hybrid(M, g, *w)))
            for name, S in {"order": M, "order+gen+cos": hybrid(M, g, *w)}.items():
                runs.setdefault(name + tag, []).append({**scored(graph, S), "w": w})
    return {name: {**mean_runs(rs), **{d: float(np.mean([r[d]["MAP"] for r in rs])) for d in ("C", "C expanded", "D", "D expanded")},
                   "identity": {k: float(np.mean([r["identity"][k] for r in rs])) for k in rs[0]["identity"]},
                   "w": [r["w"] for r in rs]} for name, rs in runs.items()}


if __name__ == "__main__":
    graph, res = SkillGraph(), load("knn_nli.json")
    P = neighbours(graph)
    print(len(P), "candidate pairs from", len(collection(graph)), "skills", file=sys.stderr)
    for key in sys.argv[1:] or ["nli"]:
        s = scores(graph, key, P)
        with torch.no_grad():
            grid = {c: row(graph, edges(graph, s, c[0], c[1]), c[2], c[3])
                    for c in itertools.product(KS, THRESHOLDS, (False, True), (False, True))}  # (k, t, soft, direct)
            k, t, soft, direct = c = max(grid, key=lambda c: grid[c]["val A MAP"])
            pairs, _ = closure(graph, edges(graph, s, k, t), direct)
            gen = generality(graph, pairs, soft)
            train_pairs = closure(graph, edges(graph, s, k, t))[0]
            direct_pairs = closure(graph, edges(graph, s, k, t), True)[0]
            res[key] = {"NLI s": s["s"], "pairs": len(P), "ms/pair": 1000 * s["s"] / (2 * len(P)),
                        "config (k, t, soft, direct)": c, "training positives": len(train_pairs),
                        "cos+gen": {**grid[c], **scored(graph, cosine_generality(graph.X, gen, grid[c]["λ"]))},
                        "best per k": {kk: grid[max((c for c in grid if c[0] == kk), key=lambda c: grid[c]["val A MAP"])] for kk in KS},
                        "grid": {str(c): {"A": r["A"]["MAP"], "B": r["B"]["MAP"], "val A": r["val A MAP"], "edges": r["edges"]}
                                 for c, r in grid.items()}}
        res[key] |= order_runs(graph, nli_graph(graph, train_pairs, gen))
        res[key] |= order_runs(graph, nli_graph(graph, train_pairs, gen, direct_pairs), " (direct positives)")
        save("knn_nli.json", res)
        r = res[key]
        print(f"{key}: {r['ms/pair']:.1f} ms/pair, config {c}, {r['training positives']} training positives", file=sys.stderr)
        for name in ("cos+gen", "order", "order+gen+cos", "order (direct positives)", "order+gen+cos (direct positives)"):
            x = r[name]
            C, CE, D, DE = (x[k]["MAP"] if isinstance(x[k], dict) else x[k] for k in ("C", "C expanded", "D", "D expanded"))
            print(f"  {name:34} A {x['A']['MAP']:.3f}  B {x['B']['MAP']:.3f}  C {C:.3f}  C expanded {CE:.3f}  D {D:.3f}  D expanded {DE:.3f}  identity "
                  + " ".join(f"{v:.3f}" for v in x["identity"].values()), file=sys.stderr)
        for kk, x in r["best per k"].items(): print(f"  cos+gen best at k={kk}: A {x['A']['MAP']:.3f}  B {x['B']['MAP']:.3f}", file=sys.stderr)
