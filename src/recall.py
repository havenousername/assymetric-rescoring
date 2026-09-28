"""Recall at prefetch depth, generalization and training cost per method. Test split, 3 seeds, configs from the sweeps.
R@K = share of a query's true answers in its top K of the 925 training skills; K = 10 / 50 / 100 / 200 ≈ 1 / 5 / 11 / 22%.
Serving is prefetch top K + rerank, so R@K at the prefetch depth caps anything a reranker can reach.
no-evidence hit@10: Task B pairs where no graph neighbour of the query has the answer (probe.py), i.e. not graph-derivable.
train s: wall time on this machine (M5 Pro, MPS), including (λ, μ) tuning for +gen+cos rows.
Usage: uv run python src/recall.py → results/recall.json"""
import time
from probe import *

cells = {k: [p for p in pairs if STRATA[k](p)] for k in ("no nb", "not named, no nb")}
res = {}

def flat(r, h, secs):
    out = {f"{t} {k}": float(v) for t in ("A", "B") if t in r for k, v in r[t].items() if k != "n"}
    return {**out, **{k: float(np.mean([h[p] for p in c])) for k, c in cells.items()}, "train s": secs}

def run(make, seeds=(0, 1, 2), ind=True):
    """make(seed) → {row: score matrix}, trained once per seed; the wall time counts toward each of its rows."""
    rows = {}
    for seed in seeds:
        torch.manual_seed(seed); random.seed(seed); t = time.time()
        Ms = make(seed)
        if dev == "mps": torch.mps.synchronize()
        secs = time.time() - t
        with torch.no_grad():
            for name, M in Ms.items(): rows.setdefault(name, []).append(flat(evaluate(M, "test", node_task=ind), hits10(M), secs))
    for name, rs in rows.items():
        res[name] = {k: float(np.mean([r[k] for r in rs])) for k in rs[0]}
        print(name, "done", file=sys.stderr)
    json.dump(res, open("results/recall.json", "w"), indent=1)

def plus_hybrid(name, M):
    with torch.no_grad(): w = max(HW, key=lambda w: val_score(hybrid(M, *w)))
    return {name: M, name + "+gen+cos": hybrid(M, *w)}

def folded(seed):
    teacher, w = box_teacher()
    return {"distilled dual, folded": distill(teacher, dim=64, tau=1.0, epochs=300, fold=w)}

if __name__ == "__main__":
    with torch.no_grad(): lam = max((0.01, 0.03, 0.1, 0.3, 1), key=lambda l: evaluate(vote_pop(l), "val", node_task=False)["B"]["MAP"])
    run(lambda s: {"cosine": cos_matrix, "pop": pop, "cos+gen": cosgen_matrix(0.05)}, seeds=(0,))
    run(lambda s: {"vote": vote, "vote+pop": vote_pop(lam)}, seeds=(0,), ind=False)
    run(lambda s: {"hyp (free vectors)": trained("hyp_trans", s)}, ind=False)
    run(lambda s: {"dual": trained("dual", s)})
    for key in ("box", "order", "hyp", "transe"): run(lambda s, key=key: plus_hybrid(key, trained(key, s)))
    run(lambda s: {"hyp+isa": trained("hyp+isa", s)})
    run(lambda s: {"pair_mlp": trained("pair_mlp", s)})
    run(folded)
    cols = ["A MAP", "A R@10", "A R@50", "A R@100", "A R@200", "B MAP", "B R@10", "B R@50", "B R@100", "B R@200",
            "no nb", "not named, no nb", "train s"]
    print(f"{'model':24}" + "".join(f"{c:>9}" for c in ["A MAP", "R@10", "R@50", "R@100", "R@200", "B MAP", "R@10", "R@50",
                                                        "R@100", "R@200", "no-evid", "no-ev,unn", "train s"]))
    for name, r in res.items(): print(f"{name:24}" + "".join(f"{r[c]:9.3f}" if c in r else f"{'–':>9}" for c in cols))
