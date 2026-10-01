"""Two diagnostics for the local graph, through the same training path as experiments/knn_nli.py (nli_graph +
trained heads, (λ, μ) on Claude's val, 3 seeds, tasks A–D and identity):
1. locality: order and order + gen + cos trained on Claude's train edges restricted to the cosine top-10 pairs (clean
   edges, local candidates); control = all of Claude's train edges through the same path. Restricted ≈ control →
   the local graph's problem is judge noise; restricted ≈ cos + gen → it is candidate locality.
2. box: box and box + gen + cos trained on the kNN → NLI graph (large NLI, k = 10, t = 0.3, direct edges), the local
   graph only order was trained on so far.
Usage: uv run python -m experiments.locality [locality|box ...] → results/locality.json"""
import json
import sys

import numpy as np
import torch

from skillmatch.data import ROOT, SkillGraph
from skillmatch.methods import hybrid
from skillmatch.methods.hybrid import GRID
from skillmatch.tasks import val_score

from .knn_nli import SEEDS, edges, neighbours, nli_graph, scored, scores
from .local_graph import knn_pairs, report, stats
from .nollm_gen import closure, generality
from .zoo import load, mean_runs, save, trained


def claude_edges(graph):
    return [(a, b, 1.0) for a, b in json.loads((ROOT / "data" / "graph.json").read_text())["train_edges"]]


def restricted(graph, E):
    knn = knn_pairs(graph)
    return [(a, b, p) for a, b, p in E if tuple(sorted((graph.index[a], graph.index[b]))) in knn]


def training_graph(graph, E):
    """The local pipeline's training copy: gen from the direct edges, positives = their closure."""
    gen = generality(graph, closure(graph, E, True)[0])
    return nli_graph(graph, closure(graph, E)[0], gen)


def head_runs(graph, g, head):
    """knn_nli.order_runs for any head."""
    runs = {}
    for seed in SEEDS:
        _, M = trained(g, head, seed)
        with torch.no_grad():
            w = max(GRID, key=lambda w: val_score(graph, hybrid(M, g, *w)))
            for name, S in {head: M, head + "+gen+cos": hybrid(M, g, *w)}.items():
                runs.setdefault(name, []).append({**scored(graph, S), "w": w})
    return {name: {**mean_runs(rs), **{d: float(np.mean([r[d]["MAP"] for r in rs])) for d in ("C", "C expanded", "D", "D expanded")},
                   "identity": {k: float(np.mean([r["identity"][k] for r in rs])) for k in rs[0]["identity"]},
                   "w": [r["w"] for r in rs]} for name, rs in runs.items()}


def show(tag, r):
    m = lambda v: v["MAP"] if isinstance(v, dict) else v
    for name, x in r.items():
        if not isinstance(x, dict) or "A" not in x: continue
        print(f"{tag:28} {name:16} A {x['A']['MAP']:.3f}  B {x['B']['MAP']:.3f}  C exp {m(x['C expanded']):.3f}  D exp {m(x['D expanded']):.3f}  "
              f"identity up {x['identity']['up top 1']:.3f} down {x['identity']['down direct top 1']:.3f}", file=sys.stderr)


if __name__ == "__main__":
    graph, res = SkillGraph(), load("locality.json")
    for what in sys.argv[1:] or ("locality", "box"):
        if what == "locality":
            E = claude_edges(graph)
            for tag, edges_ in (("Claude edges, all", E), ("Claude edges, kNN top-10 only", restricted(graph, E))):
                with torch.no_grad(): cg = report(graph, edges_, 0.0)
                res[tag] = {**stats(graph, edges_), "cos+gen": cg, **head_runs(graph, training_graph(graph, edges_), "order")}
                print(f"{tag}: {len(edges_)} edges, cos+gen A {cg['A']['MAP']:.3f} B {cg['B']['MAP']:.3f}", file=sys.stderr)
                show(tag, res[tag]); save("locality.json", res)
        else:
            E = edges(graph, scores(graph, "nli", neighbours(graph)), 10, 0.3)
            res["box on kNN→NLI graph"] = head_runs(graph, training_graph(graph, E), "box")
            show("kNN→NLI graph", res["box on kNN→NLI graph"]); save("locality.json", res)
