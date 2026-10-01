"""Round 1: cosine, cos + gen, dual encoder, box embeddings (text and free vectors) and box + gen + cos.
Test split, 3 seeds (mean, MAP std), configs picked on val by --sweep.
Usage: uv run python -m experiments.spike [--sweep] → results/spike.json (--sweep → results/sweep_best.json)"""
import itertools
import json
import sys

import torch

from skillmatch.data import SkillGraph
from skillmatch.methods import cosine, cosine_generality, hybrid
from skillmatch.methods.hybrid import tune
from skillmatch.tasks import evaluate, fit_direction, val_score

from .zoo import config, fit, mean_runs, print_table, save, seed_all

LAMBDAS = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2]  # cos + gen, picked on val A MAP
SWEEP = {"dual": {"dim": [32, 64, 128]}, "box": {"dim": [32, 64, 128], "beta_i": [0.01, 0.1, 0.3]}}
SCHEDULES = list(itertools.product([20, 60, 150], [1e-3, 3e-3]))  # (epochs, lr)


def sweep(graph):
    best = {}
    for name, grid in SWEEP.items():
        for combo in itertools.product(*grid.values()):
            kw = dict(zip(grid, combo))
            for epochs, lr in SCHEDULES:
                seed_all(0)
                _, M = fit(graph, name, kw, epochs, lr)
                with torch.no_grad(): score = val_score(graph, M)
                print(f"{name} {kw} ep={epochs} lr={lr}: val {score:.3f}", file=sys.stderr)
                if score > best.get(name, (-1,))[0]: best[name] = (score, kw, epochs, lr)
    save("sweep_best.json", best)
    return best


def main(graph):
    results = {}
    with torch.no_grad():
        results["cosine"] = evaluate(graph, cosine(graph.X), "test")
        lam = max(LAMBDAS, key=lambda l: evaluate(graph, cosine_generality(graph.X, graph.generality, l), "val")["A"]["MAP"])
        results[f"cos+gen(λ={lam})"] = evaluate(graph, cosine_generality(graph.X, graph.generality, lam), "test")
    runs = {}
    for seed in (0, 1, 2):
        seed_all(seed)
        for key in ("dual", "box", "box_trans"):
            _, M = fit(graph, key, *config(key))
            with torch.no_grad():
                r = evaluate(graph, M, "test", node_task=key != "box_trans")
                r["train_fit_dir"] = fit_direction(graph, M)
                runs.setdefault(key, []).append(r)
                if key == "box":
                    w = tune(graph, M)
                    runs.setdefault("box+gen+cos", []).append({**evaluate(graph, hybrid(M, graph, *w), "test"), "w": list(w)})
        print("seed", seed, "done", file=sys.stderr)
    for name, rs in runs.items():
        results[name] = mean_runs(rs)
        if "w" in rs[0]: results[name]["w"] = [r["w"] for r in rs]
    save("spike.json", results)
    print_table(results)


if __name__ == "__main__":
    graph = SkillGraph()
    if "--sweep" in sys.argv: print(json.dumps(sweep(graph), indent=1))
    else: main(graph)
