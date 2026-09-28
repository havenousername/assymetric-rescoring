"""Down direction: requirement → who has it (Task A transposed), the direction a "who knows FP?" search runs.
Task D: every training skill b that some unseen test skill implies ranks all 1035 candidates (925 training + 110 held-out test
skills). Relevant: held-out test skills that imply b. Filtered: b itself and training skills that imply b (known, or via
held-out edges) are removed, so the score measures finding unseen profiles among the rest of the catalogue.
Same models, configs, seeds and (λ, μ) as recall.py. (λ, μ) were tuned on val for the up direction; in the down direction
λ·gen(b) is constant for a query, so only μ·cos changes the ranking and cos+gen ranks exactly like cosine.
Free vectors, vote and pop can't take part: unseen skills have no vector or edges.
"(expanded)" rows score the ingest-time route on the same queries: each profile ranks the catalogue in the tested up
direction, and a requirement ranks profiles by where it falls in their list (store top K → match if rank ≤ K).
Usage: uv run python src/down.py → results/down.json"""
from recall import *

def down(M, split="test"):
    qa = [ix[q] for q in g["heldout_nodes"][split]]
    rel = {}
    for a in qa:
        for b in anc_full[a] & set(tr_list): rel.setdefault(b, set()).add(a)
    known = {b: {a for a in tr_list if b in anc_full[a]} | {b} for b in rel}  # b's own profile is a match, not a miss
    return rank_metrics(lambda b, a: M(a, b).T, sorted(rel), tr_list + qa, rel, known)

def expanded(M):
    """Ingest-time expansion as a down scorer: profile a scores −(rank of b among a's implied catalogue skills).
    Profiles sharing a rank tie (rank_metrics breaks ties at random). Breaking them by the raw score scored lower:
    cos+gen 0.273 vs 0.32, i.e. the direct down score is worse than chance among tied profiles."""
    col = {c: j for j, c in enumerate(tr_list)}
    def S(a, b):
        rank = M(a, torch.tensor(tr_list)).argsort(-1, descending=True).argsort(-1).float()
        return -rank[:, [col[x] for x in b.tolist()]]
    return S

if __name__ == "__main__":
    runs = [(lambda s: {"cosine": cos_matrix, "cos+gen": cosgen_matrix(0.05)}, (0,)), (lambda s: {"dual": trained("dual", s)}, None)]
    runs += [(lambda s, key=key: plus_hybrid(key, trained(key, s)), None) for key in ("box", "order", "hyp", "transe")]
    runs += [(lambda s, key=key: {key: trained(key, s)}, None) for key in ("hyp+isa", "pair_mlp")] + [(folded, None)]
    res = {}
    for make, seeds in runs:
        out = {}
        for seed in seeds or (0, 1, 2):
            torch.manual_seed(seed); random.seed(seed)
            for name, M in make(seed).items():
                with torch.no_grad():
                    out.setdefault(name, []).append(down(M)); out.setdefault(name + " (expanded)", []).append(down(expanded(M)))
        for name, rs in out.items():
            res[name] = {k: float(np.mean([r[k] for r in rs])) for k in rs[0]}
            print(name, "done", file=sys.stderr)
        json.dump(res, open("results/down.json", "w"), indent=1)
    up = load("results/recall.json")
    print(f"{'model':24}{'D MAP':>8}{'MRR':>8}{'R@10':>8}{'R@50':>8}{'R@100':>8}{'n':>6} | {'A MAP (up)':>10}")
    for name, r in res.items():
        print(f"{name:24}" + "".join(f"{r[k]:8.3f}" for k in ("MAP", "MRR", "R@10", "R@50", "R@100")) + f"{r['n']:6.0f} | "
              + (f"{up[name]['A MAP']:10.3f}" if name in up else f"{'–':>10}"))
