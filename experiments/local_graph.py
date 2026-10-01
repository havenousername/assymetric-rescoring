"""Implication graphs from signals that are not embedding similarity, no LLM, no Wikidata/SO, built over
knn_nli.collection (held-out skills excluded) and scored like experiments/knn_nli.py (cos + gen, λ on val A MAP,
tasks A–D, identity). Text sources, from the skill's own label/description/wiki snippet only:
- mention:  A's text names B's label (Jekyll "… written in Ruby" → Jekyll → Ruby); mutual mentions dropped.
- genus:    B's label occurs in A's definitional genus phrase ("JavaScript library for …" → React → JavaScript;
            "X is a Y …" from the wiki first sentence), light heads ("form of", "implementation of") skipped.
- label:    B's label is a whole-token substring of A's label (ASP.NET Core → ASP.NET).
- mention gen: no graph at all, gen(B) = log(1 + #collection texts naming B).
Every graph reports #edges, build seconds, share of edges outside the cosine top-10 pairs, recall of Claude's
train edges that lie outside those pairs, precision against Claude's closure (and the share reversed).
Usage: uv run python -m experiments.local_graph [source ...] → results/local_graph.json"""
import json
import math
import re
import sys
import time

import torch

from skillmatch.data import DEVICE, ROOT, SkillGraph
from skillmatch.methods import cosine_generality

from .knn_nli import collection, neighbours, scored
from .nollm_gen import closure, cos_gen, generality, row
from .zoo import load, save

LIGHT = {"form", "type", "kind", "set", "family", "collection", "suite", "part", "version", "implementation", "variant",
         "dialect", "series", "group", "class", "piece", "component", "branch", "subset", "superset", "fork", "style"}
STOP = re.compile(r"\b(for|that|which|used|written|based|developed|created|designed|by|in|with|to|on|from|and|or|of)\b|[,;:(]")


def pattern(label):
    """Whole-word match; short labels (C, R, Go) case-sensitive."""
    return re.compile(r"(?<![\w.#+-])" + re.escape(label) + r"(?![\w#+-])", 0 if len(label) < 4 else re.IGNORECASE)


def texts(graph, coll):
    return {i: f"{graph.nodes[graph.ids[i]]['desc'] or ''}. {graph.nodes[graph.ids[i]]['wiki'] or ''}" for i in coll}


def patterns(graph, coll):
    return {i: pattern(graph.label(i)) for i in coll}


def hits(graph, coll, text_of):
    """{a: {b}}: b's label occurs in text_of[a]."""
    pats = patterns(graph, coll)
    return {a: {b for b in coll if b != a and pats[b].search(text_of[a])} for a in coll}


def mention(graph):
    coll = collection(graph)
    h = hits(graph, coll, texts(graph, coll))
    return [(graph.ids[a], graph.ids[b], 1.0) for a in coll for b in h[a] if a not in h[b]]


def genus_phrase(graph, i):
    """The definitional noun phrase: the description up to its first stop word, plus the wiki "X is a …" complement."""
    n = graph.nodes[graph.ids[i]]
    out = []
    for s in (n["desc"] or "", n["wiki"] or ""):
        s = s.strip()
        m = re.match(r"^.{0,80}?\b(?:is|was|are)\s+(?:an?|the)\s+(.*)$", s.split(". ")[0]) if s is n["wiki"] else None
        if m: s = m.group(1)
        elif s is n["wiki"]: continue
        s = re.sub(r"^(an?|the)\s+", "", s, flags=re.IGNORECASE)
        while (m := re.match(r"^(?:[\w-]+\s+)*?(\w+)\s+of\s+(.*)$", s)) and m.group(1).lower() in LIGHT: s = m.group(2)
        out.append(STOP.split(s, 1)[0].strip())
    return " | ".join(out)


def genus(graph):
    coll = collection(graph)
    h = hits(graph, coll, {i: genus_phrase(graph, i) for i in coll})
    return [(graph.ids[a], graph.ids[b], 1.0) for a in coll for b in h[a] if a not in h[b]]


def label(graph):
    coll = collection(graph)
    toks = {i: graph.label(i).lower().split() for i in coll}
    sub = lambda s, t: any(t[k:k + len(s)] == s for k in range(len(t) - len(s) + 1))
    return [(graph.ids[a], graph.ids[b], 1.0) for a in coll for b in coll if a != b and len(toks[b]) < len(toks[a]) and sub(toks[b], toks[a])]


def mention_gen(graph):
    coll = collection(graph)
    h = hits(graph, coll, texts(graph, coll))
    n = [0] * len(graph.ids)
    for a in coll:
        for b in h[a]: n[b] += 1
    return torch.tensor([math.log1p(k) for k in n], device=DEVICE)


SOURCES = {"mention": mention, "genus": genus, "label": label}


def knn_pairs(graph, k=10):
    return {(i, j) for i, j, r in neighbours(graph) if r <= k}


def stats(graph, edges):
    """Edge diagnostics against the cosine top-10 pairs and Claude's graph (direct edges of the built graph, before
    closure() drops the held-out edges: "edges (train)" is the count after)."""
    knn = knn_pairs(graph)
    E = {(graph.index[a], graph.index[b]) for a, b, _ in edges}
    out_knn = [e for e in E if tuple(sorted(e)) not in knn]
    train = {(graph.index[a], graph.index[b]) for a, b in json.loads((ROOT / "data" / "graph.json").read_text())["train_edges"]}
    train_out = {e for e in train if tuple(sorted(e)) not in knn}
    return {"edges": len(E), "outside kNN": len(out_knn) / max(len(E), 1),
            "recall train edges": len(E & train) / len(train), "recall train edges outside kNN": len(E & train_out) / len(train_out),
            "precision vs closure": sum(b in graph.ancestors.get(a, ()) for a, b in E) / max(len(E), 1),
            "reversed": sum(a in graph.ancestors.get(b, ()) for a, b in E) / max(len(E), 1)}


def report(graph, edges, seconds, directs=(True, False)):
    """cos + gen with gen from the edges, direct or closure picked on val A MAP, tasks A–D and identity, edge stats."""
    grid = {d: row(graph, edges, False, d) for d in directs}
    direct = max(grid, key=lambda d: grid[d]["val A MAP"])
    gen = generality(graph, closure(graph, edges, direct)[0])
    r = grid[direct]
    return {**r, **scored(graph, cosine_generality(graph.X, gen, r["λ"])), "direct": direct, "build s": seconds, **stats(graph, edges),
            "edges (train)": r["edges"], "closure edges": grid.get(False, {}).get("edges")}


def gen_report(graph, gen, seconds):
    """A graph-free generality prior."""
    r = cos_gen(graph, gen)
    return {**r, **scored(graph, cosine_generality(graph.X, gen, r["λ"])), "build s": seconds}


def line(name, r):
    f = lambda k: f"{r[k]['MAP']:.3f}" if isinstance(r.get(k), dict) else f"{r.get(k, float('nan')):.3f}"
    ident = r["identity"]
    ok = ident["up top 1"] >= 0.99 and ident["down direct top 1"] >= 0.99 and ident["down expanded top 10"] == 1.0
    return (f"{name:36} edges {r.get('edges', '-'):>6} build {r.get('build s', 0):6.1f}s outside-kNN {r.get('outside kNN', float('nan')):.2f} "
            f"recall-out {r.get('recall train edges outside kNN', float('nan')):.2f} prec {r.get('precision vs closure', float('nan')):.2f} "
            f"| A {f('A')} B {f('B')} C exp {f('C expanded')} D exp {f('D expanded')} | identity {'pass' if ok else 'FAIL'} | λ {r['λ']}")


if __name__ == "__main__":
    graph, res = SkillGraph(), load("local_graph.json")
    with torch.no_grad():
        for name in sys.argv[1:] or [*SOURCES, "mention gen"]:
            t = time.time()
            if name == "mention gen": res[name] = gen_report(graph, mention_gen(graph), time.time() - t)
            else:
                E = SOURCES[name](graph)
                res[name] = report(graph, E, time.time() - t)
            save("local_graph.json", res)
            print(line(name, res[name]), file=sys.stderr)
