"""The search demo as one self-contained HTML page: tools/cvsearch_demo.html filled with precomputed results.

A static page can't run the models, so for each of the n_queries keywords most named in the indexed CVs (plus FORCED
and the TECH keywords) it stores, per mode and method, the top CVs that name it and the top fallback CVs, scored as
Search scores them (cvsearch.search.cv_scores), each with the keyword that gave its score, the keyword that ties it
to the request, and the graph's P for both where the graph has the pair. Keywords too common to be a requirement are
left out. The page reads requests longest match first, with the tokenizer and keyword rules of cvsearch.data.
The dataset is anonymized by its authors, but some CVs still carry profile URLs and phone numbers: scrub() replaces
URLs, emails and numbers with 10+ digits before any text goes into the page.
"""
import json
import math
import re
from collections import Counter

import networkx as nx
import torch

from .data import ROOT, TECH, key, tokens
from .search import COMMON, cv_scores

TEMPLATE = ROOT / "tools" / "cvsearch_demo.html"
EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
URL = re.compile(r"(?:https?://|www\.)\S+|\b[\w-]+(?:\.[\w-]+)*\.(?:com|net|org|io|ua|ru|me|dev|in)/\S*", re.I)
PHONE = re.compile(r"\+?\d[\d ()\-]{7,}\d")


def scrub(text):
    """URLs, emails and phone-like numbers (10+ digits) → placeholders."""
    text = URL.sub("[link]", EMAIL.sub("[email]", text))
    return PHONE.sub(lambda m: "[number]" if sum(c.isdigit() for c in m.group()) >= 10 else m.group(), text)
FORCED = ["LLVM", "Clang", "Qt", "STL", "Spring Boot", "Hibernate", "Jetpack Compose", "SwiftUI", "Django", "Laravel",
          "Kubernetes", "Redux", "Unreal Engine", "Boost"]


def _paths(G, q, reverse=False):
    """{b: P} over the best path from q (or into q when reverse), P = product of edge probabilities."""
    H = G.reverse(copy=False) if reverse else G
    d = nx.single_source_dijkstra_path_length(H, q, weight=lambda u, v, e: -math.log(e["weight"]))
    return {b: math.exp(-x) for b, x in d.items() if b != q}


@torch.no_grad()
def export(path, index, graph, scorers, names, metrics, n_queries=1000, n_named=5, n_fallback=10, text_cap=1500, seed=0):
    vocab = index.vocab
    df = Counter(k for s in index.sets for k in s)
    extra = [vocab.index.get(key(tokens(n))) for n in FORCED + TECH]
    queries = [k for k in dict.fromkeys([k for k, _ in df.most_common(n_queries)] + [k for k in extra if k is not None])
               if index.share.get(k, 0) < COMMON]
    rng = torch.Generator().manual_seed(seed)
    used_cvs, used_kw, Q = set(), set(queries), {}
    for q in queries:
        named = index.names({q})
        down_p, up_p = _paths(graph.G, q, reverse=True), _paths(graph.G, q)
        hits = named.nonzero()[:, 0]
        kw_order = hits[torch.randperm(len(hits), generator=rng)][:n_named].tolist()
        entry = {"n": int(named.sum()), "keyword": [[i, q, -1, q, -1] for i in kw_order],
                 "out": [[b, round(p, 2)] for b, p in sorted(((b, graph.G[q][b]["weight"]) for b in graph.G[q]), key=lambda x: -x[1])[:8]],
                 "in": [[a, round(graph.G[a][q]["weight"], 2)] for a in sorted(graph.G.predecessors(q), key=lambda a: -graph.G[a][q]["weight"])[:12]]}
        used_cvs.update(kw_order)
        used_kw.update(b for b, _ in entry["out"]); used_kw.update(a for a, _ in entry["in"])
        for mode in ("down", "up"):
            P = down_p if mode == "down" else up_p
            for m, M in scorers.items():
                s, because, close = cv_scores(index, M, q, mode)
                lists = []
                for mask, k in ((named, n_named), (~named, n_fallback)):
                    cand = torch.where(mask, s, torch.tensor(-math.inf))
                    top = [i for i in cand.topk(k).indices.tolist() if math.isfinite(cand[i])]
                    b, c = (lambda i: int(because[i])), (lambda i: int(close[i]))
                    lists.append([[i, b(i), round(P.get(b(i), -1), 2), c(i), round(down_p.get(c(i), -1), 2)] for i in top])
                    used_cvs.update(top); used_kw.update(b(i) for i in top); used_kw.update(c(i) for i in top)
                entry[f"{mode}:{m}"] = lists
        Q[q] = entry
    keys = {}
    for q in queries: keys[vocab.keys[q]] = q
    for v, k in vocab.variants.items():
        if vocab.index[k] in Q: keys[v] = vocab.index[k]
    payload = {
        "queries": {str(q): e for q, e in Q.items()}, "keys": keys,
        "kw": {str(k): vocab[k] for k in used_kw},
        "cvs": {str(i): [index.cvs[i]["position"], index.cvs[i]["years"], index.cvs[i]["pk"], scrub(index.cvs[i]["text"][:text_cap])] for i in used_cvs},
        "methods": [{"key": m, "name": names[m], **{k: round(v, 3) for k, v in metrics[names[m]].items()}} for m in scorers],
        "baseline": {k: round(v, 3) for k, v in metrics["keyword search"].items()},
        "tech": TECH, "indexed": len(index),
    }
    html = TEMPLATE.read_text().replace("__DATA__", json.dumps(payload, separators=(",", ":"), ensure_ascii=False).replace("</", "<\\/"))
    path.write_text(html)
    return {"queries": len(Q), "cvs": len(used_cvs), "MB": round(len(html.encode()) / 1e6, 1)}
