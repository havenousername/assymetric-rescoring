"""Serve the scorers from Qdrant. Needs a server: podman run -d -p 6333:6333 docker.io/qdrant/qdrant

Up (skill → what it implies): one point per catalogue skill (catalogue_collection). A head with a native distance
stores its vectors (dual: Dot, TransE: Manhattan); the other geometries store their representation in the payload
and a Formula Query rescores a cosine prefetch (skillmatch.formulas).

Down (requirement → who has it), ingest-time expansion: Qdrant inference (models.Document; FastEmbed inside the
client here, the same call runs server side on Qdrant Cloud) embeds each profile's text into a dense "text" vector.
A scorer expands each profile into its top K implied catalogue skills, stored as a sparse vector "<name>@<K>":
index = skill id, value = 926 − rank (top = 925). Per profile, so ingesting one CV rescores nobody.
Query: requirement text → dense search in "skills" → skill id b → sparse query {b: 1} on "profiles". Several
requirements = several ids; their values add.
"""
import torch
from qdrant_client import models

from .data import ENCODER


def document(text):
    return models.Document(text=text, model=ENCODER)


def recreate(client, name, **config):
    if client.collection_exists(name): client.delete_collection(name)
    client.create_collection(name, **config)


def catalogue_collection(client, name, graph, vectors, payload=None, batch=256):
    """One point per catalogue skill, id = node id. vectors: {vector name: (rows by node id, distance)}.
    payload: {field: rows by node id}, stored next to the label and the generality (field "gen")."""
    recreate(client, name, vectors_config={v: models.VectorParams(size=X.shape[1], distance=d) for v, (X, d) in vectors.items()})
    points = [models.PointStruct(id=b, vector={v: X[b].tolist() for v, (X, _) in vectors.items()},
                                 payload={"label": graph.label(b), "gen": graph.generality[b].item(),
                                          **{k: rows[b].tolist() for k, rows in (payload or {}).items()}})
              for b in graph.catalogue]
    for i in range(0, len(points), batch): client.upsert(name, points[i:i + batch])


def expansions(graph, M, profiles, ks):
    """Profiles → {K: [sparse vector of each profile's top K implied catalogue skills]}."""
    catalogue = torch.tensor(graph.catalogue)
    order = M(torch.tensor(profiles), catalogue).argsort(-1, descending=True).cpu()
    value = (len(catalogue) - torch.arange(len(catalogue))).tolist()  # value of the skill at each rank
    return {k: [models.SparseVector(indices=catalogue[o[:k]].tolist(), values=value[:k]) for o in order] for k in ks}


def ingest_profiles(client, graph, scorers, ks, profiles, texts, payloads, batch=256):
    """Upsert profiles (ids = node ids, or new ids past them for CVs added with graph.add_texts)."""
    vectors = [{"text": document(t)} for t in texts]
    for name, M in scorers.items():
        for k, sparse in expansions(graph, M, profiles, ks).items():
            for v, sv in zip(vectors, sparse): v[f"{name}@{k}"] = sv
    for i in range(0, len(profiles), batch):
        client.upsert("profiles", [models.PointStruct(id=p, vector=v, payload=pl)
                                   for p, v, pl in zip(profiles[i:i + batch], vectors[i:i + batch], payloads[i:i + batch])])


def build(client, graph, scorers, ks, profiles):
    """Collections "skills" (the catalogue, dense) and "profiles" (dense + one sparse expansion per scorer and K)."""
    dense = {"text": models.VectorParams(size=graph.X.shape[1], distance=models.Distance.COSINE)}
    recreate(client, "skills", vectors_config=dense)
    client.upsert("skills", [models.PointStruct(id=i, vector={"text": document(graph.text(i))},
                                                payload={"label": graph.label(i), "qid": graph.ids[i]}) for i in graph.catalogue])
    recreate(client, "profiles", vectors_config=dense,
             sparse_vectors_config={f"{n}@{k}": models.SparseVectorParams() for n in scorers for k in ks})
    catalogue = set(graph.catalogue)
    ingest_profiles(client, graph, scorers, ks, profiles, [graph.text(a) for a in profiles],
                    [{"label": graph.label(a), "kind": "catalogue" if a in catalogue else "unseen"} for a in profiles])


def link(client, requirement, k=1):
    """Requirement text → catalogue skill ids, by dense search (Qdrant inference embeds the query)."""
    return [p.id for p in client.query_points("skills", query=document(requirement), using="text", limit=k).points]


def who_has(client, skills, using, limit=10, **kw):
    """Profiles whose stored expansion contains the skills, best first. kw → query_points (e.g. query_filter)."""
    query = models.SparseVector(indices=list(skills), values=[1.0] * len(skills))
    return client.query_points("profiles", query=query, using=using, limit=limit, with_payload=True, **kw).points


def served(fetch):
    """Down scorer from Qdrant results, for tasks.down: fetch(b) → scored points. Profiles it doesn't return score
    −1e9 (tie last). Qdrant orders tied points by id; rank_metrics re-breaks ties at random, as for torch."""
    def S(a, b):
        out, row = torch.full((len(a), len(b)), -1e9), {x: i for i, x in enumerate(a.tolist())}
        for j, q in enumerate(b.tolist()):
            for p in fetch(q):
                if p.id in row: out[row[p.id], j] = p.score
        return out
    return S


def served_up(search):
    """Up scorer from Qdrant results, for tasks.evaluate: search(a) → scored catalogue points, higher = implied."""
    S = served(search)
    return lambda a, b: S(b, a).T
