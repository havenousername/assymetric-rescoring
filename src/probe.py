"""Tests for two claims that were guesses in docs/results.md. Task B test pairs, 3 seeds, sweep configs.
  1. Why is hyp_trans the best free-vector model? Candidate causes:
       curvature     → euc_trans: same model, loss and training, flat space
       N&K loss      → order_nk_trans: order geometry trained with hyp's softmax loss
       neighbourhood → vote: no learning; u inherits the ancestors of nodes with similar graph neighbourhoods
       popularity    → pop: no learning; rank every candidate by log(1 + #train descendants); vote+pop = vote with pop as backoff
  2. Why does text generalize? Hit rate split by whether the parent's label appears in the child's encoder text.
Strata per held-out pair (u, v): nb = vote(u, v) > 0 (some graph neighbour of u has ancestor v);
named = v's label occurs in text(u), the string the encoder embedded.
Usage: uv run python src/probe.py"""
import re
from compare import *

# ---- neighbour vote: Jaccard over training neighbours (ancestors ∪ descendants), votes for their ancestors ----
R = torch.zeros(len(ids), len(ids))
for w, vs in anc_tr.items(): R[w, list(vs)] = 1
Nb = ((R + R.T) > 0).float()
inter = Nb @ Nb.T
J = (inter / (Nb.sum(1)[:, None] + Nb.sum(1)[None] - inter).clamp_min(1)).fill_diagonal_(0)
V = J @ R
def vote(a, b): return V[a][:, b]
Vd = V.to(dev)
def pop(a, b): return gen[b.to(dev)].expand(len(a), -1)
def vote_pop(lam): return lambda a, b: Vd[a.to(dev)][:, b.to(dev)] + lam * gen[b.to(dev)]

# ---- per-pair filtered rank on Task B test ----
new = {}
for a, b in g["new_pairs"]["test"]: new.setdefault(ix[a], set()).add(ix[b])
qs, ci = sorted(new), {c: j for j, c in enumerate(tr_list)}
pairs = [(a, b) for a in qs for b in sorted(new[a]) if b in ci]

def hits10(M):
    """P(true parent in filtered top 10) per pair, ties broken at random: box (log P clamped at −1e-6), order (energy 0)
    and vote (counts) tie a lot, and counting ties as wins inflates them."""
    S, out = M(torch.tensor(qs), torch.tensor(tr_list)).detach().cpu(), {}
    for qi, q in enumerate(qs):
        s = S[qi].clone(); s[[ci[e] for e in anc_tr[q] | {q} if e in ci]] = -float("inf")
        for b in new[q] & set(ci):
            t = s.clone(); t[[ci[o] for o in new[q] - {b} if o in ci]] = -float("inf")  # other answers don't count
            gt, eq = int((t > t[ci[b]]).sum()), int((t == t[ci[b]]).sum())
            out[(q, b)] = min(max((10 - gt) / eq, 0.0), 1.0)
    return out

def named(u, v):
    return re.search(rf"(?<!\w){re.escape(g['nodes'][ids[v]]['label'])}(?!\w)", text(ids[u]), re.I) is not None

STRATA = {"all": lambda p: True,
          "nb":  lambda p: V[p] > 0,  "no nb": lambda p: V[p] == 0,
          "named": lambda p: named(*p), "not named": lambda p: not named(*p),
          "not named, nb": lambda p: not named(*p) and V[p] > 0, "not named, no nb": lambda p: not named(*p) and V[p] == 0}

def trained(key, seed):
    torch.manual_seed(seed); random.seed(seed)
    if key in ("box", "box_trans", "dual"):
        _, kw, ep, lr = load("results/sweep_best.json")[key.split("_")[0]]
        return train(Dual(**kw) if key == "dual" else Box(inductive=key == "box", **kw), ep, lr).matrix
    _, cfg, ep, lr = load("results/compare_sweep.json")[key]
    make, fit = GRIDS[key.split("+")[0]][:2]
    m = fit(make(**{k: v for k, v in cfg.items() if k != "alpha"}), ep, lr)
    if "alpha" in cfg: m.alpha = cfg["alpha"]
    return m.matrix

if __name__ == "__main__":
    MODELS = ["box_trans", "order_trans", "order_nk_trans", "transe_trans", "euc_trans", "hyp_trans", "box", "order", "euc", "hyp"]
    cells = {k: [p for p in pairs if f(p)] for k, f in STRATA.items()}
    print(f"{len(pairs)} Task B test pairs; strata sizes: " + ", ".join(f"{k} {len(v)}" for k, v in cells.items()))
    with torch.no_grad():
        lam = max((0.01, 0.03, 0.1, 0.3, 1), key=lambda l: evaluate(vote_pop(l), "val", node_task=False)["B"]["MAP"])
        for name, M in [("vote", vote), ("pop", pop), (f"vote+pop (λ={lam} on val)", vote_pop(lam))]:
            print(f"{name}, Task B test: {json.dumps(evaluate(M, 'test', node_task=False)['B'], default=lambda x: round(float(x), 3))}")
        rows = {"cosine": [hits10(cos_matrix)], "cos+gen": [hits10(cosgen_matrix(0.05))], "pop": [hits10(pop)],
                "vote": [hits10(vote)], "vote+pop": [hits10(vote_pop(lam))]}
    for key in MODELS:
        for seed in (0, 1, 2):
            M = trained(key, seed)
            with torch.no_grad(): rows.setdefault(key, []).append(hits10(M))
        print(key, "done", file=sys.stderr)
    print(f"\nhit@10 per stratum (mean over seeds; the first five rows are deterministic)\n{'model':15}" +
          "".join(f"{k:>18}" for k in cells))
    for key, rs in rows.items():
        h = [np.mean([np.mean([r[p] for p in c]) for r in rs]) if c else float("nan") for c in cells.values()]
        print(f"{key:15}" + "".join(f"{x:18.3f}" for x in h))
