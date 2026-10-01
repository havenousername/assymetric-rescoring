"""Round 2 on the same data, splits, negatives and metrics: order, TransE, hyperbolic (+ is-a score), its Euclidean
twin, pair MLP, their free-vector (_trans) ablations, a fine-tuned cross-encoder, and box + gen + cos distilled into a
dual encoder. +gen+cos rows: head + λ·gen + μ·cos, (λ, μ) picked on val.
Usage: uv run python -m experiments.compare sweep|test [name ...] | ce | distill [fold] | table
       sweep → results/compare_sweep.json (val grid); test, ce, distill → results/compare.json (3 seeds; ce 1 seed)"""
import itertools
import sys

import torch

from skillmatch.data import SkillGraph
from skillmatch.methods import CrossEncoder, distill, hybrid
from skillmatch.methods.llm_judge import Judged, test_tasks
from skillmatch.methods.hybrid import tune
from skillmatch.tasks import evaluate, fit_direction, fit_map, val_score
from skillmatch.training import labelled_pairs

from .zoo import box_teacher, config, fit, head_of, load, mean_runs, print_table, save, seed_all

GRIDS = {  # key: (head grid, (epochs, lr) schedules)
    "order": ({"dim": [32, 64, 128], "margin": [0.1, 1.0]}, [(60, 1e-3), (150, 1e-3), (150, 3e-3)]),
    "order_trans": ({"dim": [32, 64, 128], "margin": [0.1, 1.0]}, [(150, 1e-2), (300, 1e-2)]),
    "hyp": ({"dim": [5, 10, 32]}, [(60, 1e-3), (150, 1e-3), (150, 3e-3)]),
    "hyp_trans": ({"dim": [5, 10, 32]}, [(150, 1e-2), (300, 1e-2), (300, 3e-2)]),
    "pair_mlp": ({"hidden": [512]}, [(20, 1e-3), (60, 1e-3), (150, 1e-3), (60, 3e-3)]),
    "transe": ({"dim": [32, 128], "margin": [1.0, 4.0], "p": [1, 2]}, [(60, 1e-3), (150, 1e-3), (150, 3e-3)]),
    "transe_trans": ({"dim": [32, 128], "margin": [1.0, 4.0], "p": [1, 2]}, [(150, 1e-2), (300, 1e-2)]),
    "euc": ({"dim": [5, 10, 32, 128, 256]}, [(60, 1e-3), (150, 1e-3), (150, 3e-3)]),
    "euc_trans": ({"dim": [5, 10, 32, 128, 256]}, [(150, 1e-2), (300, 1e-2), (300, 3e-2)]),
    "order_nk_trans": ({"dim": [32, 128]}, [(150, 1e-2), (300, 1e-2), (300, 3e-2)]),
}
ALPHAS = [0, 0.1, 1, 10, 100, 1000]  # is-a strength for hyp/euc; 0 = the symmetric geodesic distance
DISTILL_GRID = {"dim": [64, 128], "tau": [0.3, 1.0, 3.0], "epochs": [150, 300]}


def is_text(key):
    return not head_of(key)[1]


def sweep(graph, keys):
    best = load("compare_sweep.json")
    for key in keys:
        grid, schedules = GRIDS[key]
        for k in [k for k in best if k.split("+")[0] == key]: del best[k]
        for combo in itertools.product(*grid.values()):
            kw = dict(zip(grid, combo))
            for epochs, lr in schedules:
                seed_all(0)
                model, M = fit(graph, key, kw, epochs, lr)
                for alpha in ALPHAS if hasattr(model, "alpha") else [None]:
                    if alpha is not None: model.alpha = alpha
                    with torch.no_grad(): score = val_score(graph, M, is_text(key))
                    name, cfg = key + ("+isa" if alpha else ""), {**kw, **({"alpha": alpha} if alpha is not None else {})}
                    print(f"{name} {cfg} ep={epochs} lr={lr}: val {score:.3f}", file=sys.stderr)
                    if score > best.get(name, [-1])[0]: best[name] = [score, cfg, epochs, lr]
    save("compare_sweep.json", best)


def run_test(graph, keys):
    res = {}
    for key in keys:
        runs, hybrids = [], []
        for seed in (0, 1, 2):
            seed_all(seed)
            _, M = fit(graph, key, *config(key))
            with torch.no_grad():
                r = evaluate(graph, M, "test", node_task=is_text(key))
                r["train_fit_dir"], r["fit_MAP"] = fit_direction(graph, M), fit_map(graph, M)
                if is_text(key) and "+" not in key:
                    w = tune(graph, M)
                    hybrids.append({**evaluate(graph, hybrid(M, graph, *w), "test"), "w": w})
            runs.append(r)
        res[key] = {**mean_runs(runs), "config": list(config(key))}
        if hybrids: res[key + "+gen+cos"] = {**mean_runs(hybrids), "w": [h["w"] for h in hybrids]}
        print(key, "done", file=sys.stderr)
    return res


def rerank50(graph, ce):
    """The cross-encoder over the same cos + gen top 50 the LLM judges rerank."""
    tasks, true = test_tasks(graph)
    both = true + [(b, a) for a, b in true]
    P = lambda a, b: torch.sigmoid(ce.pairs(torch.tensor(a), torch.tensor(b))).tolist()
    rank = {q: dict(zip(c, P([q] * len(c), c))) for q, c in tasks}
    return evaluate(graph, Judged(graph, rank, dict(zip(both, P(*map(list, zip(*both)))))), "test")


def run_cross_encoder(graph, max_epochs=3):
    """One seed; the epoch count picked on val, test reported at that epoch."""
    seed_all(0)
    ce, best = CrossEncoder([graph.text(i) for i in range(len(graph.ids))]), None
    for epoch in range(1, max_epochs + 1):
        ce.fit_epoch(labelled_pairs(graph))
        score = val_score(graph, ce)
        print(f"cross_enc epoch {epoch}: val {score:.3f}", file=sys.stderr)
        if best is None or score > best[0]:
            r = evaluate(graph, ce, "test")
            r["train_fit_dir"] = fit_direction(graph, ce)
            best = (score, epoch, r, rerank50(graph, ce))
    cfg = [{"model": "BAAI/bge-reranker-base", "lr": 2e-5, "max_len": ce.max_len}, best[1], 2e-5]
    return {"cross_enc": {**best[2], "config": cfg}, "cross_enc rerank50": {**best[3], "config": cfg}}


def run_distill(graph, fold=False):
    seed_all(0)
    (teacher, w), best = box_teacher(graph), None
    for combo in itertools.product(*DISTILL_GRID.values()):
        torch.manual_seed(0)
        kw = dict(zip(DISTILL_GRID, combo))
        _, S = distill(graph, teacher, **kw, fold=w if fold else None)
        with torch.no_grad(): score = val_score(graph, S)
        print(f"distill fold={fold} {kw}: val {score:.3f}", file=sys.stderr)
        if best is None or score > best[0]: best = (score, kw)
    runs, teachers = [], []
    for seed in (0, 1, 2):
        seed_all(seed)
        teacher, w = box_teacher(graph)
        _, S = distill(graph, teacher, **best[1], fold=w if fold else None)
        with torch.no_grad():
            r = evaluate(graph, S, "test")
            r["train_fit_dir"], r["fit_MAP"] = fit_direction(graph, S), fit_map(graph, S)
            runs.append(r)
            teachers.append(evaluate(graph, teacher, "test"))
    name = "distill: box+gen+cos → dual" + (" (gen+cos folded in)" if fold else "")
    return {name: {**mean_runs(runs), "config": best[1]}, "distill teacher (box+gen+cos, same seeds)": mean_runs(teachers)}


def table():
    print_table({**load("spike.json"), **load("compare.json"), **{f"llm:{k}": v for k, v in load("llm_judge.json").items()}})


if __name__ == "__main__":
    cmd, names = (sys.argv + ["table"])[1], sys.argv[2:]
    if cmd != "table":
        graph = SkillGraph()
        if cmd == "sweep": sweep(graph, names or GRIDS)
        else:
            if cmd == "test": new = run_test(graph, [k for k in load("compare_sweep.json") if not names or k.split("+")[0] in names])
            if cmd == "ce": new = run_cross_encoder(graph)
            if cmd == "distill": new = run_distill(graph, fold="fold" in names)
            save("compare.json", {**load("compare.json"), **new})
    table()
