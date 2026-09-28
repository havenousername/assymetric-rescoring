"""Spike: can a frozen text encoder + box head place unseen skills in the right containment?

Models (same frozen text embeddings, same training pairs/negatives):
  cosine       symmetric baseline
  cos+gen      cosine + generality prior (log #train descendants of the candidate) — cheap asymmetric baseline
  dual         two-role heads, score = src(a)·tgt(b)            (friend's two-encoder architecture)
  box          inductive Gumbel box head, score = log P(b | a)  (text → box)
  box_trans    free box per training node (no text) — fit sanity + held-out edges only
Tasks:
  A  held-out nodes: rank training nodes as ancestors of an unseen skill
  B  held-out edges: filtered ranking of ancestors that only become implied through held-out edges
Direction = fraction of true pairs with score(a→b) > score(b→a).
"""
import json, math, random, sys
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

SEED = 0
ENCODER = "BAAI/bge-base-en-v1.5"
DIM, EPOCHS, BATCH, LR = 64, 60, 512, 1e-3
GAMMA = 0.5772156649  # Euler–Mascheroni, Gumbel box volume correction
torch.manual_seed(SEED); random.seed(SEED); np.random.seed(SEED)
dev = "mps" if torch.backends.mps.is_available() else "cpu"

g = json.load(open("data/graph.json"))
_test = {tuple(p) for p in g["new_pairs"]["test"]}  # pairs implied by both val and test edges stay in test only
g["new_pairs"]["val"] = [p for p in g["new_pairs"]["val"] if tuple(p) not in _test]
ids = sorted(g["nodes"]); ix = {q: i for i, q in enumerate(ids)}
train_nodes = sorted({q for e in g["train_edges"] for q in e}, key=ix.get)
tr_idx = torch.tensor([ix[q] for q in train_nodes])
anc_tr = {ix[a]: {ix[b] for b in bs} for a, bs in g["train_ancestors"].items() if a in set(train_nodes)}
anc_full = {ix[a]: {ix[b] for b in bs} for a, bs in g["full_ancestors"].items()}
desc_tr = {i: 0 for i in range(len(ids))}
for a, bs in anc_tr.items():
    for b in bs: desc_tr[b] += 1

def text(q):
    n = g["nodes"][q]
    return f"{n['label']}. {n['desc']}. {n['wiki'][:600]}".strip()

def embed():
    path = f"data/emb_{ENCODER.replace('/', '_')}.npy"
    try: return np.load(path)
    except FileNotFoundError: pass
    from sentence_transformers import SentenceTransformer
    X = SentenceTransformer(ENCODER, device=dev).encode([text(q) for q in ids], normalize_embeddings=True,
                                                        batch_size=64, show_progress_bar=True)
    np.save(path, X); return X

X = torch.tensor(embed(), dtype=torch.float32, device=dev)
gen = torch.tensor([math.log1p(desc_tr[i]) for i in range(len(ids))], device=dev)  # generality: log(1 + #train descendants)

_st = None
def add_texts(texts):
    """Encode new texts (a CV, a free-text query) as extra rows of X and returns their indices, so text heads score them
    like any node (gen = 0: no known descendants). Free vectors can't: they have no row for them."""
    global _st
    if _st is None:
        from sentence_transformers import SentenceTransformer
        _st = SentenceTransformer(ENCODER, device=dev)
    new = torch.tensor(_st.encode(texts, normalize_embeddings=True), dtype=torch.float32, device=dev)
    X.data, gen.data = torch.cat([X, new]), torch.cat([gen, gen.new_zeros(len(texts))])  # in place: importers see it
    return list(range(len(X) - len(texts), len(X)))

# ---- training pairs: closure positives; reversed, random and cousin negatives ----
pos = [(a, b) for a, bs in anc_tr.items() for b in bs]
tr_list = tr_idx.tolist()
desc_of = {b: [a for a in anc_tr if b in anc_tr[a]] for b in range(len(ids))}
def negatives(a, b):
    out = [(b, a)]
    while True:
        r = random.choice(tr_list)
        if r != a and r not in anc_tr[a]: out.append((a, r)); break
    cousins = [c for c in desc_of[b] if c != a and c not in anc_tr[a]]
    if cousins: out.append((a, random.choice(cousins)))
    return out

# ---- models ----
def mlp(out): return nn.Sequential(nn.Linear(X.shape[1], 512), nn.GELU(), nn.Linear(512, out))

class Dual(nn.Module):
    def __init__(s, dim=DIM): super().__init__(); s.src, s.tgt = mlp(dim), mlp(dim)
    def logits(s, a, b): return (s.src(X[a]) * s.tgt(X[b])).sum(-1)
    def matrix(s, a, b): return s.src(X[a]) @ s.tgt(X[b]).T

class Box(nn.Module):
    """Gumbel boxes (Dasgupta+ 2020). score(a→b) = log P(b|a) = log vol(a∩b) − log vol(a)."""
    def __init__(s, inductive=True, dim=DIM, beta_i=0.1, beta_v=1.0):
        super().__init__(); s.inductive, s.bi, s.bv = inductive, beta_i, beta_v
        s.enc = mlp(2 * dim) if inductive else nn.Embedding(len(ids), 2 * dim)
    def box(s, i):
        h = s.enc(X[i]) if s.inductive else s.enc(i.to(dev))
        c, o = h.chunk(2, -1); o = F.softplus(o)
        return c - o, c + o
    def logvol(s, z, Z): return torch.log(F.softplus(Z - z - 2 * GAMMA * s.bv, beta=1 / s.bv) + 1e-10).sum(-1)
    def inter(s, z1, Z1, z2, Z2):
        return s.bi * torch.logaddexp(z1 / s.bi, z2 / s.bi), -s.bi * torch.logaddexp(-Z1 / s.bi, -Z2 / s.bi)
    def logits(s, a, b):  # returns log P(b|a)
        za, Za = s.box(a); zb, Zb = s.box(b)
        return (s.logvol(*s.inter(za, Za, zb, Zb)) - s.logvol(za, Za)).clamp(max=-1e-6)
    def matrix(s, a, b, chunk=64):
        za, Za = s.box(a); zb, Zb = s.box(b); out = []
        for i in range(0, len(a), chunk):
            z1, Z1 = za[i:i + chunk, None], Za[i:i + chunk, None]
            out.append((s.logvol(*s.inter(z1, Z1, zb[None], Zb[None])) - s.logvol(z1, Z1)).clamp(max=-1e-6))
        return torch.cat(out)

def loss_fn(m, a, b, y):
    if hasattr(m, "loss"): return m.loss(a, b, y)
    s = m.logits(a, b)
    if isinstance(m, Box):  # s = log P; positives −log P, negatives −log(1−P)
        return -(y * s + (1 - y) * torch.log(-torch.expm1(s) + 1e-10)).mean()
    return F.binary_cross_entropy_with_logits(s, y)

def train(m, epochs=EPOCHS, lr=LR):
    m.to(dev); opt = torch.optim.Adam(m.parameters(), lr=lr)
    for ep in range(epochs):
        data = [(a, b, 1.0) for a, b in pos] + [(u, v, 0.0) for a, b in pos for u, v in getattr(m, "negatives", negatives)(a, b)]
        random.shuffle(data)
        for i in range(0, len(data), BATCH):
            a, b, y = map(torch.tensor, zip(*data[i:i + BATCH]))
            loss = loss_fn(m, a.to(dev), b.to(dev), y.float().to(dev))
            opt.zero_grad(); loss.backward(); opt.step()
    return m.eval()

# ---- scoring wrappers: matrix(a_idx, b_idx) → [len(a), len(b)] scores for a→b ----
def cos_matrix(a, b): return X[a] @ X[b].T
def cosgen_matrix(lam): return lambda a, b: X[a] @ X[b].T + lam * gen[b]

# ---- evaluation ----
def rank_metrics(M, queries, cands, rel, exclude):
    mrr, ap, rk = [], [], {k: [] for k in (10, 50, 100, 200)}
    ci = {c: j for j, c in enumerate(cands)}
    S = M(torch.tensor(queries), torch.tensor(cands)).detach().cpu()
    for qi, q in enumerate(queries):
        s = S[qi].clone()
        for e in exclude.get(q, ()):
            if e in ci: s[ci[e]] = -float("inf")
        perm = torch.randperm(len(s), generator=torch.Generator().manual_seed(qi))  # random tie-break: candidate order can't leak
        order = perm[torch.argsort(s[perm], descending=True, stable=True)].tolist()
        hits = [k for k, j in enumerate(order) if cands[j] in rel[q]]
        if not hits: continue
        mrr.append(1 / (hits[0] + 1))
        for k in rk: rk[k].append(sum(h < k for h in hits) / len(rel[q]))
        ap.append(np.mean([(n + 1) / (h + 1) for n, h in enumerate(hits)]))
    return {"MRR": np.mean(mrr), **{f"R@{k}": np.mean(v) for k, v in rk.items()}, "MAP": np.mean(ap), "n": len(mrr)}

def direction(M, pairs):
    a, b = torch.tensor([p[0] for p in pairs]), torch.tensor([p[1] for p in pairs])
    if hasattr(M, "pairs"): f, r = M.pairs(a, b), M.pairs(b, a)  # pair scorers (cross-encoder, LLM) skip the full matrix
    else:
        f = torch.cat([M(a[i:i + 256], b[i:i + 256]).diagonal() for i in range(0, len(a), 256)])
        r = torch.cat([M(b[i:i + 256], a[i:i + 256]).diagonal() for i in range(0, len(a), 256)])
    return ((f > r).float() + 0.5 * (f == r).float()).mean().item()

def evaluate(M, split, node_task=True):
    res = {}
    if node_task:
        qs = [ix[q] for q in g["heldout_nodes"][split]]
        rel = {q: anc_full[q] & set(tr_list) for q in qs}
        qs = [q for q in qs if rel[q]]
        res["A"] = rank_metrics(M, qs, tr_list, rel, {})
        res["A"]["dir"] = direction(M, [(q, b) for q in qs for b in rel[q]])
    new = {}
    for a, b in g["new_pairs"][split]: new.setdefault(ix[a], set()).add(ix[b])
    qs = sorted(new)
    res["B"] = rank_metrics(M, qs, tr_list, new, {q: anc_tr[q] | {q} for q in qs})
    res["B"]["dir"] = direction(M, [(a, b) for a in qs for b in new[a]])
    return res

def fit_direction(M):
    return direction(M, random.sample(pos, min(3000, len(pos))))

def sweep():
    """Val-set grid for the learned models; the cos+gen λ gets the same treatment in main."""
    import itertools
    grids = {"dual": (Dual, {"dim": [32, 64, 128]}), "box": (Box, {"dim": [32, 64, 128], "beta_i": [0.01, 0.1, 0.3]})}
    best = {}
    for name, (cls, grid) in grids.items():
        for combo in itertools.product(*grid.values()):
            kw = dict(zip(grid, combo))
            for epochs, lr in itertools.product([20, 60, 150], [1e-3, 3e-3]):
                torch.manual_seed(SEED); random.seed(SEED)
                m = train(cls(**kw), epochs, lr)
                with torch.no_grad(): v = evaluate(m.matrix, "val")
                score = (v["A"]["MAP"] + v["B"]["MAP"]) / 2
                print(f"{name} {kw} ep={epochs} lr={lr}: val A={v['A']['MAP']:.3f} B={v['B']['MAP']:.3f}", file=sys.stderr)
                if score > best.get(name, (-1,))[0]: best[name] = (score, kw, epochs, lr)
    json.dump(best, open("results/sweep_best.json", "w"), indent=1)
    return best

if __name__ == "__main__" and "--sweep" in sys.argv:
    print(json.dumps(sweep(), indent=1)); sys.exit()

if __name__ == "__main__":
    print(f"{len(ids)} nodes, {len(train_nodes)} train nodes, {len(pos)} train closure pairs, device={dev}")
    results = {}
    with torch.no_grad():
        results["cosine"] = evaluate(cos_matrix, "test")
        lam = max([0.0, 0.01, 0.02, 0.05, 0.1, 0.2], key=lambda l: evaluate(cosgen_matrix(l), "val")["A"]["MAP"])
        results[f"cos+gen(λ={lam})"] = evaluate(cosgen_matrix(lam), "test")
    # tuned configs from `--sweep` (val only); 3 seeds, mean reported, std in results/spike.json
    try: best = json.load(open("results/sweep_best.json"))
    except FileNotFoundError: best = {"dual": [0, {}, EPOCHS, LR], "box": [0, {}, EPOCHS, LR]}
    def hybrid(M, lam, mu): return lambda a, b: M(a, b) + lam * gen[b] + mu * cos_matrix(a, b)
    val_avg = lambda M: (lambda v: (v["A"]["MAP"] + v["B"]["MAP"]) / 2)(evaluate(M, "val"))
    runs = {}
    for seed in (0, 1, 2):
        torch.manual_seed(seed); random.seed(seed)
        for name, m, (_, kw, ep, lr) in [("dual", Dual, best["dual"]), ("box", Box, best["box"]),
                                         ("box_trans", lambda **k: Box(inductive=False, **k), best["box"])]:
            m = train(m(**kw), ep, lr)
            with torch.no_grad():
                r = evaluate(m.matrix, "test", node_task=getattr(m, "inductive", True))
                r["train_fit_dir"] = fit_direction(m.matrix); runs.setdefault(name, []).append(r)
                if name == "box":  # box + generality prior + cosine, weights tuned on val
                    lam, mu = max(((l, u) for l in (0, 0.1, 0.3, 1) for u in (0, 1, 3, 10)),
                                  key=lambda w: val_avg(hybrid(m.matrix, *w)))
                    r = evaluate(hybrid(m.matrix, lam, mu), "test"); r["w"] = [lam, mu]
                    runs.setdefault("box+gen+cos", []).append(r)
        print("seed", seed, "done", file=sys.stderr)
    for name, rs in runs.items():
        results[name] = {t: {k: float(np.mean([r[t][k] for r in rs])) for k in rs[0][t]} for t in ("A", "B") if t in rs[0]}
        results[name]["std_MAP"] = {t: float(np.std([r[t]["MAP"] for r in rs])) for t in ("A", "B") if t in rs[0]}
        if "train_fit_dir" in rs[0]: results[name]["train_fit_dir"] = float(np.mean([r["train_fit_dir"] for r in rs]))
        if "w" in rs[0]: results[name]["w"] = [r["w"] for r in rs]
    json.dump(results, open("results/spike.json", "w"), indent=1, default=float)
    print(f"{'model':18} | {'A: MRR':>7} {'R@10':>6} {'MAP':>6} {'dir':>5} | {'B: MRR':>7} {'R@10':>6} {'MAP':>6} {'dir':>5} | fit_dir")
    for k, r in results.items():
        A = r.get("A", {}); B = r["B"]
        fa = lambda d, key: f"{d[key]:.3f}" if key in d else "  -  "
        sd = r.get("std_MAP", {})
        print(f"{k:18} | {fa(A,'MRR'):>7} {fa(A,'R@10'):>6} {fa(A,'MAP'):>6} {fa(A,'dir'):>5} | "
              f"{B['MRR']:.3f}  {B['R@10']:.3f} {B['MAP']:.3f} {B['dir']:.3f} | {r.get('train_fit_dir', float('nan')):.3f}"
              + (f" | MAP sd A {sd.get('A', float('nan')):.3f} B {sd['B']:.3f}" if sd else ""))
