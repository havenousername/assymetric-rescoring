"""The trained models behind results/*.json: each key names a head, its trainer and the config picked on val.

Keys: dual, box, order, order_nk, hyp, euc, transe, pair_mlp. "_trans" = free vectors (no text, the transductive
ablation); "+isa" = hyperbolic/Euclidean scored with the is-a term (alpha in the config).
Configs: results/sweep_best.json (dual, box; experiments/spike.py --sweep) and results/compare_sweep.json (the rest;
experiments/compare.py sweep).
"""
import json
import random

import numpy as np
import torch

from skillmatch.data import ROOT
from skillmatch.methods import (BoxEmbedding, DualEncoder, EuclideanEmbedding, OrderEmbedding, PairMLP,
                                PoincareEmbedding, TransE, distill, hybrid)
from skillmatch.methods.hybrid import tune
from skillmatch.training import train_pairwise, train_softmax

RESULTS = ROOT / "results"
HEADS = {  # name: (head, trainer)
    "dual": (DualEncoder, train_pairwise),
    "box": (BoxEmbedding, train_pairwise),
    "order": (OrderEmbedding, train_pairwise),
    "order_nk": (OrderEmbedding, train_softmax),  # ablation: order geometry, N&K softmax loss
    "hyp": (PoincareEmbedding, train_softmax),
    "euc": (EuclideanEmbedding, train_softmax),  # ablation: hyp in flat space
    "transe": (TransE, train_pairwise),
    "pair_mlp": (PairMLP, train_pairwise),
}
FOLDED = {"dim": 64, "tau": 1.0, "epochs": 300}  # picked on val (experiments/compare.py distill fold)


def load(name):
    try: return json.loads((RESULTS / name).read_text())
    except FileNotFoundError: return {}


def save(name, results):
    (RESULTS / name).write_text(json.dumps(results, indent=1, default=float))


def seed_all(seed):
    torch.manual_seed(seed)
    random.seed(seed)


def head_of(key):
    """'hyp_trans+isa' → ('hyp', free vectors?)."""
    name = key.split("+")[0]
    return name.removesuffix("_trans"), name.endswith("_trans")


def config(key):
    """(head kwargs, epochs, lr) picked on val."""
    head, _ = head_of(key)
    _, kw, epochs, lr = load("sweep_best.json")[head] if head in ("dual", "box") else load("compare_sweep.json")[key]
    return kw, epochs, lr


def fit(graph, key, kw, epochs, lr):
    """Train the key's head with these hyperparameters. Returns (model, scorer)."""
    head, free = head_of(key)
    make, train = HEADS[head]
    inputs = graph.node_ids if free else graph.X
    model = train(make(**kw, n_nodes=len(graph.ids)) if free else make(**kw), graph, epochs, lr, inputs=inputs)
    return model, model.scorer(inputs)


def trained(graph, key, seed=0):
    """Seeded, with the config picked on val. Returns (model, scorer)."""
    seed_all(seed)
    return fit(graph, key, *config(key))


def plus_hybrid(graph, name, M):
    """{name: M, name+gen+cos: M + λ·gen + μ·cos}, (λ, μ) picked on val."""
    with torch.no_grad(): w = tune(graph, M)
    return {name: M, name + "+gen+cos": hybrid(M, graph, *w)}


def box_teacher(graph):
    """box + gen + cos, the distillation teacher. Unseeded: the caller's seed. Returns (scorer, (λ, μ))."""
    _, M = fit(graph, "box", *config("box"))
    with torch.no_grad(): w = tune(graph, M)
    return hybrid(M, graph, *w), w


def folded(graph):
    """The distilled dual with gen + cos folded in: one dot product. Returns (student, scorer, (λ, μ))."""
    teacher, w = box_teacher(graph)
    student, S = distill(graph, teacher, **FOLDED, fold=w)
    return student, S, w


def mean_runs(runs):
    """Mean over seeds of evaluate() outputs, plus the MAP std and the fit diagnostics."""
    tasks = [t for t in ("A", "B") if t in runs[0]]
    out = {t: {k: float(np.mean([r[t][k] for r in runs])) for k in runs[0][t]} for t in tasks}
    out["std_MAP"] = {t: float(np.std([r[t]["MAP"] for r in runs])) for t in tasks}
    for k in ("train_fit_dir", "fit_MAP"):
        if k in runs[0]: out[k] = float(np.mean([r[k] for r in runs]))
    return out


def mean_flat(runs):
    """Mean over seeds of flat {metric: value} dicts."""
    return {k: float(np.mean([float(r[k]) for r in runs])) for k in runs[0]}


def print_table(rows):
    """rows: {name: evaluate()-style result, optionally mean_runs() extras}."""
    f = lambda d, k: f"{d[k]:.3f}" if k in d else "  -  "
    print(f"{'model':22} | {'A: MRR':>6} {'R@10':>5} {'MAP':>5} {'dir':>5} | {'B: MRR':>6} {'R@10':>5} {'MAP':>5} {'dir':>5}"
          f" | fit_dir fit_MAP | MAP sd A/B")
    for k, r in rows.items():
        A, B, sd = r.get("A", {}), r["B"], r.get("std_MAP", {})
        print(f"{k:22} | {f(A,'MRR'):>6} {f(A,'R@10'):>5} {f(A,'MAP'):>5} {f(A,'dir'):>5} | {f(B,'MRR'):>6} {f(B,'R@10'):>5}"
              f" {f(B,'MAP'):>5} {f(B,'dir'):>5} | {f(r,'train_fit_dir'):>7} {f(r,'fit_MAP'):>7} | {f(sd,'A')}/{f(sd,'B')}")
