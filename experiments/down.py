"""Down direction: requirement → who has it (Task D, skillmatch.tasks.down: unseen profiles; Task C, down_known: known
profiles, hidden edges; "[C]" rows), the direction a "who knows FP?" search runs.
Same models, configs, seeds and (λ, μ) as recall.py. (λ, μ) were tuned on val for the up direction; in the down
direction λ·gen(b) is constant for a query, so only μ·cos changes the ranking and cos+gen ranks exactly like cosine.
Free vectors, vote and pop can't take part: unseen skills have no vector or edges.
"(expanded)" rows score the ingest-time route on the same queries: each profile ranks the catalogue in the tested up
direction, and a requirement ranks profiles by where it falls in their list (store top K → match if rank ≤ K).
"+depth" rows add hyp's depth term μ·d(0, profile) (PoincareEmbedding.mu), μ picked per seed on Task D val: the only
down-tuned weight here. It is a constant in the up direction, so their expanded rows equal hyp's and hyp+gen+cos's.
Usage: uv run python -m experiments.down → results/down.json"""
import copy
import sys

import torch

from skillmatch.data import SkillGraph
from skillmatch.methods import cosine, cosine_generality, hybrid
from skillmatch.methods.hybrid import tune
from skillmatch.tasks import down, down_known, expanded

from .zoo import folded, load, mean_flat, plus_hybrid, save, seed_all, trained

SEEDS = (0, 1, 2)
DEPTH_MUS = (0, 0.5, 0.9, 1, 1.1, 1.2, 1.3, 1.5, 2)


def hyp_depth(graph, seed):
    """hyp and hyp + gen + cos (its (λ, μ) up-tuned, as in the other rows), each with its own depth μ picked on D val."""
    model, M = trained(graph, "hyp", seed)
    with torch.no_grad(): w = tune(graph, M)
    twin = copy.deepcopy(model)
    rows = {"hyp+depth": (model, M), "hyp+depth+gen+cos": (twin, hybrid(twin.scorer(graph.X), graph, *w))}
    for name, (m, S) in rows.items():
        def val(mu):
            m.mu = mu
            return down(graph, S, "val")["MAP"]
        with torch.no_grad(): m.mu = max(DEPTH_MUS, key=val)
        print(f"{name} seed {seed}: mu {m.mu}", file=sys.stderr)
    return {name: S for name, (_, S) in rows.items()}


def runs(graph, solo=("hyp+isa", "pair_mlp"), distilled=True):
    """[(make(seed) → {name: scorer}, seeds)]: the models that can score an unseen profile."""
    out = [(lambda s: {"cosine": cosine(graph.X), "cos+gen": cosine_generality(graph.X, graph.generality)}, (0,)),
           (lambda s: {"dual": trained(graph, "dual", s)[1]}, SEEDS)]
    out += [(lambda s, key=key: plus_hybrid(graph, key, trained(graph, key, s)[1]), SEEDS) for key in ("box", "order", "hyp", "transe")]
    out += [(lambda s, key=key: {key: trained(graph, key, s)[1]}, SEEDS) for key in solo]
    out.append((lambda s: hyp_depth(graph, s), SEEDS))
    if distilled: out.append((lambda s: {"distilled dual, folded": folded(graph)[1]}, SEEDS))
    return out


def collect(runs, measure, path):
    """Mean over seeds of measure(name, scorer) → {row: {metric: value}}, merged into results/<path> after every run."""
    res = load(path)
    for make, seeds in runs:
        out = {}
        for seed in seeds:
            seed_all(seed)
            for name, M in make(seed).items():
                with torch.no_grad():
                    for row, metrics in measure(name, M).items(): out.setdefault(row, []).append(metrics)
        for row, rs in out.items():
            res[row] = mean_flat(rs)
            print(row, "done", file=sys.stderr)
        save(path, res)
    return res


if __name__ == "__main__":
    graph = SkillGraph()
    res = collect(runs(graph), lambda name, M: {name: down(graph, M), name + " (expanded)": down(graph, expanded(graph, M)),
                                                name + " [C]": down_known(graph, M), name + " [C] (expanded)": down_known(graph, expanded(graph, M))}, "down.json")
    up = load("recall.json")
    print(f"{'model':24}{'D MAP':>8}{'MRR':>8}{'R@10':>8}{'R@50':>8}{'R@100':>8}{'n':>6} | {'A MAP (up)':>10}")
    for name, r in res.items():
        print(f"{name:24}" + "".join(f"{r[k]:8.3f}" for k in ("MAP", "MRR", "R@10", "R@50", "R@100")) + f"{r['n']:6.0f} | "
              + (f"{up[name]['A MAP']:10.3f}" if name in up else f"{'–':>10}"))
