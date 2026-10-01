"""Ranking metrics for any scorer M(a ids, b ids) → [len(a), len(b)] scores."""
import numpy as np
import torch


def rank_metrics(M, queries, candidates, relevant, exclude):
    """MRR, R@K, MAP over queries that have a relevant candidate. exclude[q]: candidates dropped from q's list.
    Ties are broken at random (seeded per query), so candidate order can't leak into the metric."""
    mrr, ap, recall = [], [], {k: [] for k in (10, 50, 100, 200)}
    col = {c: j for j, c in enumerate(candidates)}
    S = M(torch.tensor(queries), torch.tensor(candidates)).detach().cpu()
    for qi, q in enumerate(queries):
        s = S[qi].clone()
        for e in exclude.get(q, ()):
            if e in col: s[col[e]] = -float("inf")
        perm = torch.randperm(len(s), generator=torch.Generator().manual_seed(qi))
        order = perm[torch.argsort(s[perm], descending=True, stable=True)].tolist()
        hits = [k for k, j in enumerate(order) if candidates[j] in relevant[q]]
        if not hits: continue
        mrr.append(1 / (hits[0] + 1))
        for k in recall: recall[k].append(sum(h < k for h in hits) / len(relevant[q]))
        ap.append(np.mean([(n + 1) / (h + 1) for n, h in enumerate(hits)]))
    return {"MRR": np.mean(mrr), **{f"R@{k}": np.mean(v) for k, v in recall.items()}, "MAP": np.mean(ap), "n": len(mrr)}


def direction(M, pairs):
    """Share of true pairs (A, B) with s(A→B) > s(B→A); ties count half. Pair scorers (M.pairs: cross-encoder, LLM)
    skip the full matrix."""
    a, b = torch.tensor([p[0] for p in pairs]), torch.tensor([p[1] for p in pairs])
    if hasattr(M, "pairs"): f, r = M.pairs(a, b), M.pairs(b, a)
    else:
        f = torch.cat([M(a[i:i + 256], b[i:i + 256]).diagonal() for i in range(0, len(a), 256)])
        r = torch.cat([M(b[i:i + 256], a[i:i + 256]).diagonal() for i in range(0, len(a), 256)])
    return ((f > r).float() + 0.5 * (f == r).float()).mean().item()
