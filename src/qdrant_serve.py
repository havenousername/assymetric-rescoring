"""Serve the down direction (requirement → who has it) from Qdrant.
Ingest: Qdrant inference (models.Document: FastEmbed in the client here, the same call runs server side on Qdrant Cloud)
embeds each profile's text into a dense "text" vector; a head expands the profile into its top K implied catalogue
skills, stored as a sparse vector "<head>@<K>" (index = skill, value = 926 − rank, rank 1 = top). Per profile, so ingesting a CV
rescores nobody. Same ranking as expanded() in down.py.
Query: requirement text → dense search in "skills" → skill index b → sparse query {b: 1} on "profiles".
Several requirements = several indices; their values add.
Eval: Task D (down.py) through Qdrant vs the same ranking in torch. Profiles past K aren't returned; they tie last and
rank_metrics orders them at random, so @K metrics give a little credit to profiles Qdrant never returned.
Latency is measured on the 1035 Task D profiles.
Heads read the cached sentence-transformers rows; FastEmbed returns the same vectors (cos 1.000 on 200 nodes).
Needs a server: podman run -d -p 6333:6333 docker.io/qdrant/qdrant
Usage: uv run python src/qdrant_serve.py → results/qdrant.json"""
import time
from qdrant_client import QdrantClient, models
from down import *

qc = QdrantClient("localhost", port=6333, timeout=300)
KS = (10, 50, 100, 925)
PROFILES = tr_list + [ix[q] for q in g["heldout_nodes"]["test"]]  # Task D candidates: catalogue + unseen test skills

def label(i): return g["nodes"][ids[i]]["label"]
def doc(t): return models.Document(text=t, model=ENCODER)

def recreate(name, **kw):
    if qc.collection_exists(name): qc.delete_collection(name)
    qc.create_collection(name, **kw)

def sparse(M, a):
    """Profiles a → {K: [SparseVector of each profile's top K implied catalogue skills]}."""
    order, cat = M(torch.tensor(a), torch.tensor(tr_list)).argsort(-1, descending=True).cpu(), torch.tensor(tr_list)
    val = (len(cat) - torch.arange(len(cat))).tolist()  # value of the skill at each rank
    return {k: [models.SparseVector(indices=cat[o[:k]].tolist(), values=val[:k]) for o in order] for k in KS}

def ingest(heads, a, texts, payloads, batch=256):
    vecs = [{"text": doc(t)} for t in texts]
    for name, M in heads.items():
        for k, svs in sparse(M, a).items():
            for v, sv in zip(vecs, svs): v[f"{name}@{k}"] = sv
    for i in range(0, len(a), batch):
        qc.upsert("profiles", [models.PointStruct(id=p, vector=v, payload=pl)
                               for p, v, pl in zip(a[i:i + batch], vecs[i:i + batch], payloads[i:i + batch])])

def build(heads):
    dense = {"text": models.VectorParams(size=X.shape[1], distance=models.Distance.COSINE)}
    recreate("skills", vectors_config=dense)
    qc.upsert("skills", [models.PointStruct(id=i, vector={"text": doc(text(ids[i]))}, payload={"label": label(i), "qid": ids[i]})
                         for i in tr_list])
    recreate("profiles", vectors_config=dense,
             sparse_vectors_config={f"{n}@{k}": models.SparseVectorParams() for n in heads for k in KS})
    cat = set(tr_list)
    ingest(heads, PROFILES, [text(ids[a]) for a in PROFILES],
           [{"label": label(a), "kind": "catalogue" if a in cat else "unseen"} for a in PROFILES])

def link(req, k=1):
    """Requirement text → catalogue skill ids by dense search (Qdrant inference embeds the query)."""
    return [p.id for p in qc.query_points("skills", query=doc(req), using="text", limit=k).points]

def who_has(skills, using, limit=10, **kw):
    """kw → query_points, e.g. query_filter to keep only CVs."""
    return qc.query_points("profiles", query=models.SparseVector(indices=list(skills), values=[1.0] * len(skills)),
                           using=using, limit=limit, with_payload=True, **kw).points

def served(fetch):
    """Down scorer from Qdrant: fetch(b) → scored points; profiles it doesn't return score −1e9 (tie last).
    Qdrant orders tied points by id; rank_metrics re-breaks ties at random, as for torch."""
    def S(a, b):
        out, row = torch.full((len(a), len(b)), -1e9), {x: i for i, x in enumerate(a.tolist())}
        for j, q in enumerate(b.tolist()):
            for p in fetch(q):
                if p.id in row: out[row[p.id], j] = p.score
        return out
    return S

if __name__ == "__main__":
    n, res = len(PROFILES), {}
    order = plus_hybrid("order", trained("order", 0))["order+gen+cos"]
    with torch.no_grad():
        heads = {"order+gen+cos": order, "cos+gen": cosgen_matrix(0.05)}
        t = time.time(); build(heads); res["ingest s"] = time.time() - t
        print(f"ingest: {len(tr_list)} skills + {n} profiles in {res['ingest s']:.0f}s", file=sys.stderr)
        res["dense, direct"] = down(served(lambda b: qc.query_points("profiles", query=doc(text(ids[b])), using="text", limit=n).points))
        for name, M in heads.items():
            res[f"{name}, torch"] = down(expanded(M))
            for k in KS: res[f"{name}@{k}"] = down(served(lambda b, u=f"{name}@{k}": who_has([b], u, n)))
        # end to end: the requirement's label as free text → linked skill → sparse lookup
        linked, lat = {}, []
        def e2e(b, u="order+gen+cos@100"):
            t = time.perf_counter(); s = link(label(b)); who_has(s, u, 10); lat.append(time.perf_counter() - t)
            linked[b] = s[0]
            return who_has(s, u, n)
        res["order+gen+cos@100, label text linked"] = down(served(e2e))
        res["link@1"] = float(np.mean([linked[b] == b for b in linked]))
        res["p50 ms (link + top 10)"] = 1000 * float(np.median(lat))
    json.dump(res, open("results/qdrant.json", "w"), indent=1, default=float)
    print(f"{'route':40}{'D MAP':>8}{'R@10':>8}{'R@50':>8}{'R@100':>8}{'n':>6}")
    for name, r in res.items():
        if isinstance(r, dict): print(f"{name:40}" + "".join(f"{r[k]:8.3f}" for k in ("MAP", "R@10", "R@50", "R@100")) + f"{r['n']:6.0f}")
        else: print(f"{name:40}{r:8.3f}")
