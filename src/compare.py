"""Round 2 on the spike's data, splits, negatives and metrics:
  hyp        Poincaré ball, as in the Qdrant hyperbolic article (Mahmood & Kupchanko 2026), trained with the
             Nickel & Kiela 2017 loss (the article gives no training details). Distance is symmetric.
  hyp+isa    same embedding, scored with N&K's is-a score −(1 + α(‖b‖ − ‖a‖))·d(a, b); α tuned on val
  order      order embeddings (Vendrov+ 2016): a ⊑ b iff f(a) ≥ f(b) coordinate-wise
  pair_mlp   pairwise classifier on frozen text embeddings, MLP([a, b, a⊙b, a−b])
  cross_enc  fine-tuned BAAI/bge-reranker-base reading (A text, B text)
  transe     TransE with one relation: score(a→b) = −‖f(a) + r − f(b)‖; query f(a) + r → native Euclid/Manhattan HNSW
  euc        ablation of hyp: identical model, loss and training, Euclidean distance instead of Poincaré
  order_nk   ablation: order embeddings trained with hyp's N&K softmax loss instead of max-margin
  distill    box+gen+cos teacher → dual student (listwise KL over training nodes); serves as named vectors + Dot HNSW
*_trans: a free vector per node, no text (the article's setup); held-out edges only.
+gen+cos: model score + λ·generality + μ·cosine, (λ, μ) tuned on val (as box+gen+cos in spike.py).
Usage: uv run python src/compare.py sweep|test [name ...] | ce | distill [fold] | table
       (sweep = val grid; test = 3 seeds; ce = cross-encoder; distill = sweep + 3 seeds)
"""
import itertools
from spike import *

# ---- hyperbolic ----
def expmap0(v):  # tangent space at the origin → Poincaré ball (curvature −1)
    n = v.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    return torch.tanh(n) * v / n

def poincare(u, v):
    uu, vv = (u * u).sum(-1).clamp(max=1 - 1e-5), (v * v).sum(-1).clamp(max=1 - 1e-5)
    x = (1 + 2 * ((u - v) ** 2).sum(-1) / ((1 - uu) * (1 - vv))).clamp_min(1 + 1e-7)
    return torch.log(x + torch.sqrt(x * x - 1))  # acosh, written as in the article's Formula Query

class Hyper(nn.Module):
    """score(a→b) = −(1 + α(‖b‖ − ‖a‖))·d(a, b). α = 0 is the article's symmetric geodesic distance."""
    def __init__(s, inductive=True, dim=5, alpha=0.0):
        super().__init__(); s.inductive, s.alpha = inductive, alpha
        s.enc = mlp(dim) if inductive else nn.Embedding(len(ids), dim)
        if not inductive: nn.init.uniform_(s.enc.weight, -1e-3, 1e-3)  # N&K init near the origin
    def x(s, i): return expmap0(s.enc(X[i]) if s.inductive else s.enc(i.to(dev)))
    def dist(s, xa, xb): return poincare(xa, xb)
    def score(s, xa, xb): return -(1 + s.alpha * (xb.norm(dim=-1) - xa.norm(dim=-1))) * s.dist(xa, xb)
    def logits(s, a, b): return s.score(s.x(a), s.x(b))
    def matrix(s, a, b):
        xa, xb = s.x(a), s.x(b)
        return torch.cat([s.score(xa[i:i + 64, None], xb[None]) for i in range(0, len(a), 64)])

class Euc(Hyper):
    """Ablation: Hyper with flat space — no exp map, Euclidean distance. Isolates what the curvature contributes."""
    def x(s, i): return s.enc(X[i]) if s.inductive else s.enc(i.to(dev))
    def dist(s, xa, xb): return ((xa - xb) ** 2).sum(-1).clamp_min(1e-12).sqrt()

related = {a: anc_tr[a] | set(desc_of[a]) | {a} for a in tr_list}
def unrelated(a, k):
    out = []
    while len(out) < k:
        r = random.choice(tr_list)
        if r not in related[a]: out.append(r)
    return out

def train_nk(m, epochs, lr, K=10, batch=64):
    """Nickel & Kiela 2017: softmax over −d, true ancestor vs K nodes unrelated to the query in either direction.
    Any model with x(i) and dist(xa, xb). ponytail: Adam on tangent vectors via expmap0 instead of Riemannian SGD
    + burn-in; switch if fit stalls."""
    m.to(dev); opt = torch.optim.Adam(m.parameters(), lr=lr)
    for _ in range(epochs):
        P = random.sample(pos, len(pos))
        for i in range(0, len(P), batch):
            u, v = zip(*P[i:i + batch])
            c = torch.tensor([[b] + unrelated(a, K) for a, b in zip(u, v)], device=dev)
            d = m.dist(m.x(torch.tensor(u, device=dev))[:, None], m.x(c))
            loss = F.cross_entropy(-d, torch.zeros(len(u), dtype=torch.long, device=dev))
            opt.zero_grad(); loss.backward(); opt.step()
    return m.eval()

# ---- order embeddings ----
class Order(nn.Module):
    """Vendrov+ 2016. f ≥ 0, specific skills far from the origin; score(a→b) = −‖max(0, f(b) − f(a))‖², 0 iff a ⊑ b."""
    def __init__(s, inductive=True, dim=DIM, margin=1.0):
        super().__init__(); s.inductive, s.margin = inductive, margin
        s.enc = mlp(dim) if inductive else nn.Embedding(len(ids), dim)
    def f(s, i): return (s.enc(X[i]) if s.inductive else s.enc(i.to(dev))).abs()
    def energy(s, fa, fb): return F.relu(fb - fa).pow(2).sum(-1)
    def logits(s, a, b): return -s.energy(s.f(a), s.f(b))
    def matrix(s, a, b):
        fa, fb = s.f(a), s.f(b)
        return torch.cat([-s.energy(fa[i:i + 64, None], fb[None]) for i in range(0, len(a), 64)])
    def loss(s, a, b, y):  # Vendrov's max-margin: E on positives, max(0, margin − E) on negatives
        E = -s.logits(a, b); return (y * E + (1 - y) * F.relu(s.margin - E)).mean()
    def x(s, i): return s.f(i)  # for train_nk: energy as the distance
    def dist(s, fa, fb): return s.energy(fa, fb)

class TransE(Order):
    """Bordes+ 2013 with one relation r: energy ‖f(a) + r − f(b)‖_p, same max-margin loss and negatives as Order."""
    def __init__(s, inductive=True, dim=DIM, margin=1.0, p=2):
        super().__init__(inductive, dim, margin); s.p, s.r = p, nn.Parameter(0.1 * torch.randn(dim))
    def f(s, i): return s.enc(X[i]) if s.inductive else s.enc(i.to(dev))
    def energy(s, fa, fb): return (fa + s.r - fb).norm(p=s.p, dim=-1)

# ---- pairwise scorers (serving: rerank only) ----
class PairMLP(nn.Module):
    """MLP([a, b, a⊙b, a−b]) of frozen text embeddings → logit P(b|a). Order-sensitive, hence asymmetric."""
    def __init__(s, hidden=512):
        super().__init__(); d = X.shape[1]
        s.net = nn.Sequential(nn.Linear(4 * d, hidden), nn.GELU(), nn.Linear(hidden, 1))
    def f(s, xa, xb): return s.net(torch.cat([xa, xb, xa * xb, xa - xb], -1)).squeeze(-1)
    def logits(s, a, b): return s.f(X[a], X[b])
    def matrix(s, a, b):
        xb = X[b]
        return torch.cat([s.f(X[a[i:i + 16]][:, None].expand(-1, len(b), -1), xb[None].expand(len(a[i:i + 16]), -1, -1))
                          for i in range(0, len(a), 16)])

class CrossEnc:
    """Fine-tuned cross-encoder on (A text, B text) → logit P(B|A)."""
    def __init__(s, name="BAAI/bge-reranker-base", max_len=96):
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        s.tok, s.max_len, s.T = AutoTokenizer.from_pretrained(name), max_len, [text(q) for q in ids]
        s.model = AutoModelForSequenceClassification.from_pretrained(name).to(dev)
    def logits(s, a, b):
        enc = s.tok([s.T[i] for i in a], [s.T[j] for j in b], truncation=True, max_length=s.max_len,
                    padding=True, return_tensors="pt").to(dev)
        return s.model(**enc).logits.squeeze(-1)
    def pairs(s, a, b, bs=256):
        s.model.eval()  # fp16 inference: 2.5× faster on MPS, logits within 0.06 of fp32
        with torch.no_grad(), torch.autocast(dev, dtype=torch.float16, enabled=dev == "mps"):
            return torch.cat([s.logits(a[i:i + bs].tolist(), b[i:i + bs].tolist()).float() for i in range(0, len(a), bs)])
    def __call__(s, a, b): return s.pairs(a.repeat_interleave(len(b)), b.repeat(len(a))).view(len(a), len(b))
    def fit_epoch(s, lr=2e-5, bs=32):
        if not hasattr(s, "opt"): s.opt = torch.optim.AdamW(s.model.parameters(), lr=lr)
        data = [(a, b, 1.0) for a, b in pos] + [(u, v, 0.0) for a, b in pos for u, v in negatives(a, b)]
        random.shuffle(data); s.model.train()
        for i in range(0, len(data), bs):
            a, b, y = zip(*data[i:i + bs])
            loss = F.binary_cross_entropy_with_logits(s.logits(a, b), torch.tensor(y, device=dev))
            s.opt.zero_grad(); loss.backward(); s.opt.step()

# ---- runs ----
GRIDS = {  # name: (constructor, trainer, model grid, (epochs, lr) schedule)
    "order":       (Order, train, {"dim": [32, 64, 128], "margin": [0.1, 1.0]}, [(60, 1e-3), (150, 1e-3), (150, 3e-3)]),
    "order_trans": (lambda **k: Order(inductive=False, **k), train, {"dim": [32, 64, 128], "margin": [0.1, 1.0]},
                    [(150, 1e-2), (300, 1e-2)]),
    "hyp":         (Hyper, train_nk, {"dim": [5, 10, 32]}, [(60, 1e-3), (150, 1e-3), (150, 3e-3)]),
    "hyp_trans":   (lambda **k: Hyper(inductive=False, **k), train_nk, {"dim": [5, 10, 32]},
                    [(150, 1e-2), (300, 1e-2), (300, 3e-2)]),
    "pair_mlp":    (PairMLP, train, {"hidden": [512]}, [(20, 1e-3), (60, 1e-3), (150, 1e-3), (60, 3e-3)]),
    "transe":      (TransE, train, {"dim": [32, 128], "margin": [1.0, 4.0], "p": [1, 2]}, [(60, 1e-3), (150, 1e-3), (150, 3e-3)]),
    "transe_trans": (lambda **k: TransE(inductive=False, **k), train, {"dim": [32, 128], "margin": [1.0, 4.0], "p": [1, 2]},
                     [(150, 1e-2), (300, 1e-2)]),
    "euc":         (Euc, train_nk, {"dim": [5, 10, 32, 128, 256]}, [(60, 1e-3), (150, 1e-3), (150, 3e-3)]),
    "euc_trans":   (lambda **k: Euc(inductive=False, **k), train_nk, {"dim": [5, 10, 32, 128, 256]},
                    [(150, 1e-2), (300, 1e-2), (300, 3e-2)]),
    "order_nk_trans": (lambda **k: Order(inductive=False, **k), train_nk, {"dim": [32, 128]},
                       [(150, 1e-2), (300, 1e-2), (300, 3e-2)]),
}
ALPHAS = [0, 0.1, 1, 10, 100, 1000]  # N&K is-a strength; 0 = the article's symmetric distance
def hybrid(M, lam, mu): return lambda a, b: M(a, b) + lam * gen[b] + mu * cos_matrix(a, b)
HW = [(l, u) for l in (0, 0.1, 0.3, 1) for u in (0, 1, 3, 10)]  # same (λ, μ) grid as box+gen+cos in spike.py

def inductive(name): return not name.split("+")[0].endswith("_trans")
def val_score(M, ind=True):
    v = evaluate(M, "val", node_task=ind)
    return (v["A"]["MAP"] + v["B"]["MAP"]) / 2 if ind else v["B"]["MAP"]

def fit_map(M, n=300):
    """Directed MAP on training pairs (cf. the article's MAP on seen relationships)."""
    qs = random.Random(0).sample([q for q in tr_list if anc_tr[q]], n)
    return rank_metrics(M, qs, tr_list, {q: anc_tr[q] for q in qs}, {q: {q} for q in qs})["MAP"]

def sweep(names):
    best = load("results/compare_sweep.json")
    for name in names:
        make, fit, grid, sched = GRIDS[name]
        for k in [k for k in best if k.split("+")[0] == name]: del best[k]
        for combo in itertools.product(*grid.values()):
            kw = dict(zip(grid, combo))
            for ep, lr in sched:
                torch.manual_seed(SEED); random.seed(SEED)
                m = fit(make(**kw), ep, lr)
                for al in ALPHAS if isinstance(m, Hyper) else [None]:
                    if al is not None: m.alpha = al
                    with torch.no_grad(): sc = val_score(m.matrix, inductive(name))
                    key, cfg = name + ("+isa" if al else ""), {**kw, **({"alpha": al} if al is not None else {})}
                    print(f"{key} {cfg} ep={ep} lr={lr}: val {sc:.3f}", file=sys.stderr)
                    if sc > best.get(key, [-1])[0]: best[key] = [sc, cfg, ep, lr]
    json.dump(best, open("results/compare_sweep.json", "w"), indent=1)
    return best

def mean_runs(rs):
    out = {t: {k: float(np.mean([r[t][k] for r in rs])) for k in rs[0][t]} for t in ("A", "B") if t in rs[0]}
    out["std_MAP"] = {t: float(np.std([r[t]["MAP"] for r in rs])) for t in ("A", "B") if t in rs[0]}
    for k in ("train_fit_dir", "fit_MAP"):
        if k in rs[0]: out[k] = float(np.mean([r[k] for r in rs]))
    return out

def run_test(best):
    res = {}
    for key, (_, cfg, ep, lr) in best.items():
        make, fit = GRIDS[key.split("+")[0]][:2]; rs, hs = [], []
        for seed in (0, 1, 2):
            torch.manual_seed(seed); random.seed(seed)
            m = fit(make(**{k: v for k, v in cfg.items() if k != "alpha"}), ep, lr)
            if "alpha" in cfg: m.alpha = cfg["alpha"]
            with torch.no_grad():
                r = evaluate(m.matrix, "test", node_task=inductive(key))
                r["train_fit_dir"], r["fit_MAP"] = fit_direction(m.matrix), fit_map(m.matrix)
                if inductive(key) and "+" not in key:
                    w = max(HW, key=lambda w: val_score(hybrid(m.matrix, *w)))
                    hs.append({**evaluate(hybrid(m.matrix, *w), "test"), "w": w})
            rs.append(r)
        res[key] = {**mean_runs(rs), "config": [cfg, ep, lr]}
        if hs: res[key + "+gen+cos"] = {**mean_runs(hs), "w": [h["w"] for h in hs]}
        print(key, "done", file=sys.stderr)
    return res

def rerank50(ce):
    """Cross-encoder over the same cos+gen top-50 the LLM judges rerank."""
    from llm_judge import test_tasks, Judged
    tasks, true = test_tasks(); both = true + [(b, a) for a, b in true]
    P = lambda a, b: torch.sigmoid(ce.pairs(torch.tensor(a), torch.tensor(b))).tolist()
    rank = {q: dict(zip(c, P([q] * len(c), c))) for q, c in tasks}
    return evaluate(Judged(rank, dict(zip(both, P(*map(list, zip(*both)))))), "test")

def run_cross_encoder(max_epochs=3):
    """One seed; epoch count picked on val, test reported at that epoch."""
    torch.manual_seed(SEED); random.seed(SEED)
    ce, best = CrossEnc(), None
    for ep in range(1, max_epochs + 1):
        ce.fit_epoch(); sc = val_score(ce)
        print(f"cross_enc epoch {ep}: val {sc:.3f}", file=sys.stderr)
        if best is None or sc > best[0]:
            r = evaluate(ce, "test"); r["train_fit_dir"] = fit_direction(ce); best = (sc, ep, r, rerank50(ce))
    cfg = [{"model": "BAAI/bge-reranker-base", "lr": 2e-5, "max_len": ce.max_len}, best[1], 2e-5]
    return {"cross_enc": {**best[2], "config": cfg}, "cross_enc rerank50": {**best[3], "config": cfg}}

# ---- F: structured teacher → dual student ----
def box_teacher():
    """box+gen+cos as in spike.py: box config from its sweep, (λ, μ) tuned on val."""
    _, kw, ep, lr = load("results/sweep_best.json")["box"]
    m = train(Box(**kw), ep, lr)
    with torch.no_grad(): w = max(HW, key=lambda w: val_score(hybrid(m.matrix, *w)))
    return hybrid(m.matrix, *w), w

def distill(teacher, dim=64, tau=1.0, epochs=150, lr=1e-3, batch=64, fold=None):
    """Listwise KL: the student's softmax over all training nodes matches softmax(teacher / τ). Queries and candidates
    are training nodes only, so held-out nodes and edges never reach the student. fold=(λ, μ): student = src·tgt +
    λ·gen + μ·cos, so it only learns the box part; still one dot product: [src(a), √μ·x(a), 1]·[tgt(b), √μ·x(b), λ·gen(b)].
    Returns the student's score matrix function."""
    with torch.no_grad():
        T = torch.cat([teacher(tr_idx[i:i + 256], tr_idx) for i in range(0, len(tr_idx), 256)])
        P = F.softmax(T.fill_diagonal_(-float("inf")) / tau, -1)
    m = Dual(dim).to(dev); opt = torch.optim.Adam(m.parameters(), lr=lr)
    S = m.matrix if fold is None else hybrid(m.matrix, *fold)
    for _ in range(epochs):
        for q in torch.randperm(len(tr_idx)).split(batch):
            loss = F.kl_div(F.log_softmax(S(tr_idx[q], tr_idx), -1), P[q.to(dev)], reduction="batchmean")
            opt.zero_grad(); loss.backward(); opt.step()
    return S

def run_distill(fold=False):
    torch.manual_seed(SEED); random.seed(SEED); (teacher, w), best = box_teacher(), None
    for dim, tau, ep in itertools.product([64, 128], [0.3, 1.0, 3.0], [150, 300]):
        torch.manual_seed(SEED); kw = {"dim": dim, "tau": tau, "epochs": ep}
        S = distill(teacher, **kw, fold=w if fold else None)
        with torch.no_grad(): sc = val_score(S)
        print(f"distill fold={fold} {kw}: val {sc:.3f}", file=sys.stderr)
        if best is None or sc > best[0]: best = (sc, kw)
    rs, ts = [], []
    for seed in (0, 1, 2):
        torch.manual_seed(seed); random.seed(seed); teacher, w = box_teacher()
        S = distill(teacher, **best[1], fold=w if fold else None)
        with torch.no_grad():
            r = evaluate(S, "test"); r["train_fit_dir"], r["fit_MAP"] = fit_direction(S), fit_map(S)
            rs.append(r); ts.append(evaluate(teacher, "test"))
    name = "distill: box+gen+cos → dual" + (" (gen+cos folded in)" if fold else "")
    return {name: {**mean_runs(rs), "config": best[1]}, "distill teacher (box+gen+cos, same seeds)": mean_runs(ts)}

def load(p):
    try: return json.load(open(p))
    except FileNotFoundError: return {}

def save(new):
    res = load("results/compare.json"); res.update(new)
    json.dump(res, open("results/compare.json", "w"), indent=1, default=float)

def table():
    rows = {**load("results/spike.json"), **load("results/compare.json"),
            **{f"llm:{k}": v for k, v in load("results/llm_judge.json").items()}}
    f = lambda d, k: f"{d[k]:.3f}" if k in d else "  -  "
    print(f"{'model':22} | {'A: MRR':>6} {'R@10':>5} {'MAP':>5} {'dir':>5} | {'B: MRR':>6} {'R@10':>5} {'MAP':>5} {'dir':>5}"
          f" | fit_dir fit_MAP | MAP sd A/B")
    for k, r in rows.items():
        A, B, sd = r.get("A", {}), r["B"], r.get("std_MAP", {})
        print(f"{k:22} | {f(A,'MRR'):>6} {f(A,'R@10'):>5} {f(A,'MAP'):>5} {f(A,'dir'):>5} | {f(B,'MRR'):>6} {f(B,'R@10'):>5}"
              f" {f(B,'MAP'):>5} {f(B,'dir'):>5} | {f(r,'train_fit_dir'):>7} {f(r,'fit_MAP'):>7} | {f(sd,'A')}/{f(sd,'B')}")

if __name__ == "__main__":
    cmd, names = (sys.argv + ["table"])[1], sys.argv[2:]
    if cmd == "sweep": sweep(names or GRIDS)
    if cmd == "test":
        save(run_test({k: v for k, v in load("results/compare_sweep.json").items() if not names or k.split("+")[0] in names}))
    if cmd == "ce": save(run_cross_encoder())
    if cmd == "distill": save(run_distill(fold="fold" in names))
    table()
