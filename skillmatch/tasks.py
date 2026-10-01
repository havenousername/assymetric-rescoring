"""The evaluation tasks, for any scorer M(a ids, b ids) → [len(a), len(b)] scores of s(A→B).

A, up, new skill: each held-out skill ranks the 925 catalogue skills for what it implies.
B, up, known skill: each catalogue skill ranks the catalogue for implications hidden from the training graph;
   the ones the training graph already gives (and the skill itself) are filtered out.
C, down, known profile: B read the other way. Each catalogue skill b that a hidden pair implies ranks the 925
   catalogue profiles by who has b. Relevant: profiles a with a → b hidden. Filtered: b itself and the profiles the
   training graph already gives (graph lookup answers those).
D, down: each catalogue skill b that some held-out skill implies ranks 1035 profiles (catalogue + held-out skills)
   by who has b. Relevant: held-out skills that imply b. Filtered: b itself and catalogue skills that imply b.
"""
import random

import torch

from .metrics import direction, rank_metrics


def up_new(graph, split):
    """Task A: {held-out skill: its catalogue ancestors}, skills without any dropped."""
    catalogue = set(graph.catalogue)
    relevant = {q: graph.ancestors[q] & catalogue for q in graph.heldout[split]}
    return {q: r for q, r in relevant.items() if r}


def up_known(graph, split):
    """Task B: {catalogue skill: its hidden ancestors}."""
    relevant = {}
    for a, b in graph.hidden_pairs[split]: relevant.setdefault(a, set()).add(b)
    return dict(sorted(relevant.items()))


def evaluate(graph, M, split, node_task=True, directions=True):
    """Tasks A and B plus direction accuracy. node_task=False skips A (free vectors can't embed unseen skills).
    directions=False skips direction accuracy, which needs s(B→A) for held-out A: a served index has no such rows."""
    res = {}
    if node_task:
        rel = up_new(graph, split)
        res["A"] = rank_metrics(M, list(rel), graph.catalogue, rel, {})
        if directions: res["A"]["dir"] = direction(M, [(q, b) for q in rel for b in rel[q]])
    rel = up_known(graph, split)
    res["B"] = rank_metrics(M, list(rel), graph.catalogue, rel, {q: graph.ancestors_train[q] | {q} for q in rel})
    if directions: res["B"]["dir"] = direction(M, [(a, b) for a in rel for b in rel[a]])
    return res


def val_score(graph, M, node_task=True):
    """What the sweeps maximise: mean val MAP of A and B (B alone for free vectors)."""
    v = evaluate(graph, M, "val", node_task)
    return (v["A"]["MAP"] + v["B"]["MAP"]) / 2 if node_task else v["B"]["MAP"]


def down_known(graph, M, split="test"):
    """Task C, direct: profile a scores requirement b by s(a→b)."""
    catalogue, relevant = set(graph.catalogue), {}
    for a, b in graph.hidden_pairs[split]:
        if b in catalogue: relevant.setdefault(b, set()).add(a)  # 1 pair per split implies a held-out skill
    known = {b: {a for a in graph.catalogue if b in graph.ancestors_train[a]} | {b} for b in relevant}
    return rank_metrics(lambda b, a: M(a, b).T, sorted(relevant), graph.catalogue, relevant, known)


def down(graph, M, split="test"):
    """Task D, direct: profile a scores requirement b by s(a→b)."""
    profiles = graph.heldout[split]
    catalogue = set(graph.catalogue)
    relevant = {}
    for a in profiles:
        for b in graph.ancestors[a] & catalogue: relevant.setdefault(b, set()).add(a)
    known = {b: {a for a in graph.catalogue if b in graph.ancestors[a]} | {b} for b in relevant}  # b's own profile is a match
    return rank_metrics(lambda b, a: M(a, b).T, sorted(relevant), graph.catalogue + profiles, relevant, known)


def expanded(graph, M):
    """Ingest-time expansion as a down scorer: profile a scores b by −(rank of b in a's list of implied catalogue
    skills). Profiles sharing a rank tie. Breaking ties by the raw score scored lower (cos+gen 0.273 vs 0.32)."""
    catalogue = torch.tensor(graph.catalogue)
    col = {c: j for j, c in enumerate(graph.catalogue)}

    def S(a, b):
        rank = M(a, catalogue).argsort(-1, descending=True).argsort(-1).float()
        return -rank[:, [col[x] for x in b.tolist()]]
    return S


def self_rank(S):
    """Row i: rank of column i (1 = top); columns tied with it count half."""
    s = S.gather(1, torch.arange(len(S))[:, None])
    return (1 + (S > s).sum(1) + ((S == s).sum(1) - 1) / 2).float()


def identity(graph, M, split="test"):
    """Reflexivity: every skill implies itself. Up: where b lands in its own implied list (Qdrant stores the top K,
    so past K b's own profile never comes back for b). Down: where b's own profile lands among the Task D profiles."""
    catalogue = torch.tensor(graph.catalogue)
    profiles = torch.tensor(graph.catalogue + graph.heldout[split])  # first 925 profiles = catalogue, same order
    up, direct = self_rank(M(catalogue, catalogue).cpu()), self_rank(M(profiles, catalogue).T.cpu())
    via = self_rank(expanded(graph, M)(profiles, catalogue).T.cpu())
    return {"up top 1": (up == 1).float().mean(), "up median rank": up.median(), "up in top 100": (up <= 100).float().mean(),
            "down direct top 1": (direct == 1).float().mean(), "down expanded top 1": (via == 1).float().mean(),
            "down expanded top 10": (via <= 10).float().mean()}


def fit_direction(graph, M):
    """Direction accuracy on 3000 training pairs: did the head fit the direction it was shown?"""
    return direction(M, random.sample(graph.positives, min(3000, len(graph.positives))))


def fit_map(graph, M, n=300):
    """Directed MAP on training pairs (cf. the hyperbolic article's MAP on seen relationships)."""
    qs = random.Random(0).sample([q for q in graph.catalogue if graph.ancestors_train[q]], n)
    return rank_metrics(M, qs, graph.catalogue, {q: graph.ancestors_train[q] for q in qs}, {q: {q} for q in qs})["MAP"]
