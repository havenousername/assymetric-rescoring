"""Candidate pairs that do not come from the embedding, judged by the local NLI cross-encoder as in
experiments/knn_nli.py (deberta-v3-large zeroshot, "person" template, both directions, A → B kept when p ≥ t and
p(A→B) > p(B→A), t and direct/closure picked on val A MAP). Candidate sets over knn_nli.collection:
- text:      mention ∪ genus ∪ label pairs (experiments/local_graph.py);
- hubsH:     every skill × the H most general skills, generality = mention count (no graph, no embedding);
- ghubsH:    the same with generality = genus count (how many skills name it in their definitional genus phrase):
             mention-count hubs include common words (form, list, display, make) the NLI then accepts as parents;
- knn10:     the cosine top-10 pairs (the knn_nli baseline, same judge, for reference);
- unions:    text+hubs50, text+knn10, text+hubs50+knn10.
"top-m" rows keep, per skill A, only its m strongest parents (p(A→B) ≥ t, NLI direction) instead of every pair over t:
per-node selection, Chen, Lin & Klein 2021 (m, t on val A MAP); uses the same cached scores.
Usage: uv run python -m experiments.hub_nli [set ...] → results/hub_nli.json; NLI scores cached per pair in
results/hub_nli_scores.json (seeded from results/knn_nli_scores.json)."""
import sys
import time

import torch

from skillmatch.data import SkillGraph

from .knn_nli import THRESHOLDS, collection, neighbours, text
from .local_judge import NLIS, nli_scorer
from .local_graph import SOURCES, genus, line, mention_gen, report
from .zoo import load, save

JUDGE = "nli"


def text_pairs(graph):
    return {tuple(sorted((graph.index[a], graph.index[b]))) for f in SOURCES.values() for a, b, _ in f(graph)}


def hub_pairs(graph, h):
    coll = collection(graph)
    gen = mention_gen(graph)
    hubs = sorted(coll, key=lambda i: -gen[i].item())[:h]
    return {tuple(sorted((i, j))) for i in coll for j in hubs if i != j}


def genus_hub_pairs(graph, h):
    coll, n = collection(graph), {}
    for _, b, _ in genus(graph): n[graph.index[b]] = n.get(graph.index[b], 0) + 1
    hubs = sorted(n, key=lambda i: -n[i])[:h]
    return {tuple(sorted((i, j))) for i in coll for j in hubs if i != j}


def knn10_pairs(graph):
    return {(i, j) for i, j, r in neighbours(graph) if r <= 10}


SETS = {"text": text_pairs, "hubs50": lambda g: hub_pairs(g, 50), "knn10": knn10_pairs,
        "ghubs50": lambda g: genus_hub_pairs(g, 50), "text+ghubs50+knn10": lambda g: text_pairs(g) | genus_hub_pairs(g, 50) | knn10_pairs(g),
        "text+hubs50": lambda g: text_pairs(g) | hub_pairs(g, 50), "text+knn10": lambda g: text_pairs(g) | knn10_pairs(g),
        "text+hubs50+knn10": lambda g: text_pairs(g) | hub_pairs(g, 50) | knn10_pairs(g)}


def judged(graph, pairs, key=JUDGE):
    """{(i, j): (p(i→j), p(j→i))} for i < j, scored once per pair and cached. Returns (scores, seconds spent now)."""
    cache = load("hub_nli_scores.json")
    C = cache.setdefault(key, {})
    if not C and (k := load("knn_nli_scores.json").get(key)):
        C.update({f"{i},{j}": [f, b] for (i, j, _), f, b in zip(k["pairs"], k["fwd"], k["rev"])})
    todo = sorted(p for p in pairs if f"{p[0]},{p[1]}" not in C)
    t = time.time()
    if todo:
        score = nli_scorer(NLIS[key], "person")
        xy = [(text(graph, i), text(graph, j)) for i, j in todo]
        fwd, rev = score(xy), score([(y, x) for x, y in xy])
        C.update({f"{i},{j}": [f, b] for (i, j), f, b in zip(todo, fwd, rev)})
        save("hub_nli_scores.json", cache)
    return {p: tuple(C[f"{p[0]},{p[1]}"]) for p in pairs}, time.time() - t, len(todo)


def edges(graph, S, t):
    out = []
    for (i, j), (f, b) in S.items():
        a, c, p = (i, j, f) if f > b else (j, i, b)
        if p >= t: out.append((graph.ids[a], graph.ids[c], p))
    return out


def topm_edges(graph, S, m, t):
    out = {}
    for (i, j), (f, b) in S.items():
        a, c, p = (i, j, f) if f > b else (j, i, b)
        if p >= t: out.setdefault(a, []).append((p, c))
    return [(graph.ids[a], graph.ids[c], p) for a, cs in out.items() for p, c in sorted(cs, reverse=True)[:m]]


def run_topm(graph, name, ms=(1, 2, 3, 5)):
    t0 = time.time()
    P = SETS[name](graph)
    S, nli_s, _ = judged(graph, P)
    build = time.time() - t0
    rows = {(m, t): report(graph, topm_edges(graph, S, m, t), build) for m in ms for t in (0.1, 0.3, 0.5)}
    best = max(rows, key=lambda k: rows[k]["val A MAP"])
    return {**rows[best], "m, t": best, "pairs": len(P),
            "per m, t": {str(k): {"A": r["A"]["MAP"], "B": r["B"]["MAP"], "val A": r["val A MAP"], "edges": r["edges"], "prec": r["precision vs closure"]} for k, r in rows.items()}}


def run(graph, name, seconds_extra=0.0):
    t0 = time.time()
    P = SETS[name](graph)
    S, nli_s, scored_now = judged(graph, P)
    build = time.time() - t0
    rows = {t: report(graph, edges(graph, S, t), build) for t in THRESHOLDS}
    best = max(rows, key=lambda t: rows[t]["val A MAP"])
    return {**rows[best], "t": best, "pairs": len(P), "NLI s (new pairs)": nli_s, "new pairs": scored_now,
            "per t": {t: {"A": r["A"]["MAP"], "B": r["B"]["MAP"], "val A": r["val A MAP"], "edges": r["edges"], "direct": r["direct"]} for t, r in rows.items()}}


if __name__ == "__main__":
    graph, res = SkillGraph(), load("hub_nli.json")
    with torch.no_grad():
        for name in sys.argv[1:] or SETS:
            if name.endswith(" top-m"):
                res[name] = run_topm(graph, name.removesuffix(" top-m"))
                print(line(name, res[name]), "| m, t", res[name]["m, t"], file=sys.stderr)
            else:
                res[name] = run(graph, name)
                r = res[name]
                print(line(name, r), f"| t {r['t']} pairs {r['pairs']} new {r['new pairs']} NLI {r['NLI s (new pairs)']:.0f}s", file=sys.stderr)
            save("hub_nli.json", res)
