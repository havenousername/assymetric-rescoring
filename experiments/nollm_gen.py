"""cos + gen without an LLM: gen = log(1 + #descendants) counted on a graph of unjudged candidate edges
(pipeline/build_candidates.py evidence), scored on the Claude-labelled tasks A and B (skillmatch/tasks.py).
Held-out skills and held-out edges are removed first and hidden pairs never count, as for the Claude graph.
Descendants are counted over every candidate skill, not only the 925 catalogue skills. λ picked on val A MAP.
Judged rows keep candidates a judge accepts: Claude's y (same construction, the reference) and each local judge
in results/local_judge_scores.json (experiments/local_judge.py) at p(A→B) ≥ t, t picked on val A MAP;
"soft" counts each descendant by its best-path product of p instead of 1.
Usage: uv run python -m experiments.nollm_gen → results/nollm_gen.json"""
import json
import math

import networkx as nx
import numpy as np
import torch

from skillmatch.data import DEVICE, ROOT, SkillGraph
from skillmatch.methods import cosine_generality
from skillmatch.tasks import evaluate

from .local_judge import pairs
from .zoo import load, print_table, save

LAMBDAS = [0.0, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
THRESHOLDS = [0.1, 0.3, 0.5, 0.7, 0.9]
SOURCES = {  # which candidate edges count: weight(candidate id, evidence), 0 = dropped
    "wd subclass/instance": lambda i, ev: bool({"wd:subclass_of", "wd:instance_of"} & set(ev)),
    "wd any": lambda i, ev: any(e.startswith("wd:") for e in ev),
    "SO rule": lambda i, ev: any(e.startswith("so:") for e in ev),
    "all candidates": lambda i, ev: True,
}


def closure(graph, edges, direct=False):
    """{(a, b): P(A → B)} for skill-graph nodes b, from (specific, general, p) qid triples, with held-out skills,
    held-out edges and hidden pairs removed. P = best product of p along a path (1 for unjudged edges);
    direct=True keeps the edges only, no transitivity."""
    raw = json.loads((ROOT / "data" / "graph.json").read_text())
    W = nx.DiGraph()
    W.add_weighted_edges_from((a, b, -math.log(p)) for a, b, p in edges)
    W.remove_nodes_from([q for qs in raw["heldout_nodes"].values() for q in qs])
    W.remove_edges_from([tuple(e) for es in raw["held_edges"].values() for e in es])
    hidden = {p for ps in graph.hidden_pairs.values() for p in ps}
    out = {}
    for a in W:
        reach = {b: W[a][b]["weight"] for b in W[a]} if direct else nx.single_source_dijkstra_path_length(W, a)
        for b, d in reach.items():
            if b != a and b in graph.index and (graph.index.get(a), graph.index[b]) not in hidden: out[a, b] = math.exp(-d)
    return out, W


def generality(graph, pairs, soft=False):
    """log(1 + #descendants), or log(1 + Σ P(descendant → b)) when soft."""
    n = [0.0] * len(graph.ids)
    for (_, b), p in pairs.items(): n[graph.index[b]] += p if soft else 1
    return torch.tensor([math.log1p(k) for k in n], device=DEVICE)


def candidate_edges(weight):
    """(src, dst, p) for the candidates with weight(candidate id, evidence) = p > 0, duplicate-label qids merged as
    in pipeline/build_graph.py."""
    d = json.loads((ROOT / "data" / "candidates.json").read_text())
    N, canon = d["nodes"], {}
    for q in sorted(N, key=lambda q: -N[q]["so_count"]): canon.setdefault(N[q]["label"].lower(), q)
    c = lambda q: canon[N[q]["label"].lower()]
    return [(c(e["src"]), c(e["dst"]), float(w)) for i, e in enumerate(d["candidates"])
            if (w := weight(i, e["evidence"])) and c(e["src"]) != c(e["dst"])]


def cos_gen(graph, gen):
    val = {l: evaluate(graph, cosine_generality(graph.X, gen, l), "val")["A"]["MAP"] for l in LAMBDAS}
    lam = max(val, key=val.get)
    return {**evaluate(graph, cosine_generality(graph.X, gen, lam), "test"), "λ": lam, "val A MAP": val[lam]}


def row(graph, edges, soft=False, direct=False):
    pairs, W = closure(graph, edges, direct)
    return {**cos_gen(graph, generality(graph, pairs, soft)), "edges": W.number_of_edges(),
            "largest SCC": max(map(len, nx.strongly_connected_components(W)))}


if __name__ == "__main__":
    graph = SkillGraph()
    raw = json.loads((ROOT / "data" / "graph.json").read_text())
    assert torch.equal(generality(graph, closure(graph, [(a, b, 1.0) for a, b in raw["train_edges"]])[0]), graph.generality)
    popularity = torch.tensor([math.log1p(graph.nodes[q]["so_count"]) for q in graph.ids], device=DEVICE)
    results = {}
    with torch.no_grad():
        results["Claude graph"] = {**cos_gen(graph, graph.generality), "edges": len(raw["train_edges"])}
        for name, keep in SOURCES.items(): results[name] = row(graph, candidate_edges(keep))
        y = [p[3] for p in pairs()]
        results["all candidates, Claude y"] = row(graph, candidate_edges(lambda i, ev: y[i]))
        for name, s in load("local_judge_scores.json").items():
            fwd = np.array(s["fwd"])
            for soft in (False, True):
                runs = {t: row(graph, candidate_edges(lambda i, ev: fwd[i] * (fwd[i] >= t)), soft) for t in THRESHOLDS}
                t = max(runs, key=lambda t: runs[t]["val A MAP"])
                results[f"all candidates, {name}" + ", soft" * soft] = {**runs[t], "t": t}
        results["SO popularity (no graph)"] = cos_gen(graph, popularity)
    save("nollm_gen.json", results)
    print_table(results)
    for k, r in results.items(): print(f"{k:34} λ={r['λ']} t={r.get('t', '-')} edges={r.get('edges', '-')}")
