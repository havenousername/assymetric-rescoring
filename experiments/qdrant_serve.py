"""Task D served from Qdrant (skillmatch.qdrant) vs the same ranking in torch, plus the end-to-end route and latency.
Profiles past K aren't returned; they tie last and rank_metrics orders them at random, so @K metrics give a little
credit to profiles Qdrant never returned. Latency is measured on the 1035 Task D profiles.
Heads read the cached sentence-transformers rows; FastEmbed returns the same vectors (cos 1.000 on 200 nodes).
Needs a server: podman run -d -p 6333:6333 docker.io/qdrant/qdrant
Usage: uv run python -m experiments.qdrant_serve → results/qdrant.json"""
import time

import numpy as np
import torch
from qdrant_client import QdrantClient

from skillmatch import qdrant
from skillmatch.data import SkillGraph
from skillmatch.methods import cosine_generality
from skillmatch.tasks import down, expanded

from .zoo import plus_hybrid, save, trained

KS = (10, 50, 100, 925)

if __name__ == "__main__":
    graph, client = SkillGraph(), QdrantClient("localhost", port=6333, timeout=300)
    profiles = graph.catalogue + graph.heldout["test"]  # Task D candidates: catalogue + unseen test skills
    n, res = len(profiles), {}
    order = plus_hybrid(graph, "order", trained(graph, "order", 0)[1])["order+gen+cos"]
    with torch.no_grad():
        scorers = {"order+gen+cos": order, "cos+gen": cosine_generality(graph.X, graph.generality)}
        t = time.time(); qdrant.build(client, graph, scorers, KS, profiles); res["ingest s"] = time.time() - t
        print(f"ingest: {len(graph.catalogue)} skills + {n} profiles in {res['ingest s']:.0f}s")
        dense = lambda b: client.query_points("profiles", query=qdrant.document(graph.text(b)), using="text", limit=n).points
        res["dense, direct"] = down(graph, qdrant.served(dense))
        for name, M in scorers.items():
            res[f"{name}, torch"] = down(graph, expanded(graph, M))
            for k in KS: res[f"{name}@{k}"] = down(graph, qdrant.served(lambda b, u=f"{name}@{k}": qdrant.who_has(client, [b], u, n)))
        linked, latency = {}, []

        def end_to_end(b, using="order+gen+cos@100"):
            """The requirement's label as free text → linked skill → sparse lookup."""
            t = time.perf_counter()
            skills = qdrant.link(client, graph.label(b))
            qdrant.who_has(client, skills, using, 10)
            latency.append(time.perf_counter() - t)
            linked[b] = skills[0]
            return qdrant.who_has(client, skills, using, n)
        res["order+gen+cos@100, label text linked"] = down(graph, qdrant.served(end_to_end))
        res["link@1"] = float(np.mean([linked[b] == b for b in linked]))
        res["p50 ms (link + top 10)"] = 1000 * float(np.median(latency))
    save("qdrant.json", res)
    print(f"{'route':40}{'D MAP':>8}{'R@10':>8}{'R@50':>8}{'R@100':>8}{'n':>6}")
    for name, r in res.items():
        if isinstance(r, dict): print(f"{name:40}" + "".join(f"{r[k]:8.3f}" for k in ("MAP", "R@10", "R@50", "R@100")) + f"{r['n']:6.0f}")
        else: print(f"{name:40}{r:8.3f}")
