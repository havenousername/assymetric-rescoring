"""Distributional inclusion (Weeds & Weir 2003; Clarke 2009; Lenci & Benotto 2012 invCL; Santus et al. 2014 SLQS) over
two sparse feature spaces built from the collection's own texts, no embedding, no model:
- words:      TF-IDF unigrams of label + description + wiki snippet;
- comention:  binary skill × document matrix, document d is a feature of skill B when d's text names B
              (experiments/local_graph.py hits); B ⊃ A when the documents naming A also name B.
Measures on every ordered collection pair: WeedsPrec(A,B) = Σ_{f∈A∩B} w_A(f) / Σ_f w_A(f);
ClarkeDE(A,B) = Σ_{f∈A∩B} min(w_A, w_B) / Σ_f w_A(f); invCL(A,B) = sqrt(CDE(A,B)·(1 − CDE(B,A))).
Graphs: for each A its top-m partners by invCL, kept as A → B when invCL(A,B) > invCL(B,A) (m on val), unjudged.
SLQS generality (no graph): E(B) = median entropy of B's top-N features, entropy over the documents a feature occurs in.
Usage: uv run python -m experiments.dih → results/dih.json"""
import math
import sys
import time

import numpy as np
import torch
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

from skillmatch.data import DEVICE, SkillGraph

from .knn_nli import collection
from .local_graph import gen_report, hits, line, report, texts
from .zoo import load, save

MS = (3, 5, 10)


def word_features(graph, coll):
    T = texts(graph, coll)
    docs = [f"{graph.label(i)}. {T[i]}" for i in coll]
    return TfidfVectorizer(stop_words="english", sublinear_tf=True, min_df=2).fit_transform(docs).tocsr()


def comention_features(graph, coll):
    h = hits(graph, coll, texts(graph, coll))
    pos = {i: r for r, i in enumerate(coll)}
    rows, cols = zip(*[(pos[b], pos[a]) for a in coll for b in h[a] | {a}])  # skill b has feature "document a"
    return csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(coll), len(coll)))


def measures(F):
    """(WeedsPrec, ClarkeDE, invCL) dense [n, n], row = A, column = B."""
    F = F.toarray().astype(np.float32)
    total = F.sum(1, keepdims=True) + 1e-9
    shared = (F > 0).astype(np.float32)
    weeds = (F @ shared.T) / total  # Σ_{f ∈ A ∩ B} w_A(f) / Σ w_A
    cde = np.stack([np.minimum(F[i], F).sum(1) for i in range(len(F))]) / total
    invcl = np.sqrt(cde * (1 - cde.T))
    return weeds, cde, invcl


def slqs_gen(graph, coll, F, n=20):
    """log-median entropy of a skill's top-n features; entropy of a feature over the documents it weights."""
    F = F.toarray().astype(np.float64)
    P = F / (F.sum(0, keepdims=True) + 1e-12)
    H = -(P * np.log(P + 1e-12)).sum(0)  # per feature
    g = torch.zeros(len(graph.ids), device=DEVICE)
    for r, i in enumerate(coll):
        top = np.argsort(-F[r])[:n]
        top = top[F[r][top] > 0]
        g[i] = float(np.median(H[top])) if len(top) else 0.0
    return g


def edges_topm(graph, coll, M, m):
    out = []
    for r, a in enumerate(coll):
        row = M[r].copy(); row[r] = -1
        for c in np.argsort(-row)[:m]:
            if row[c] > M[c, r] and row[c] > 0: out.append((graph.ids[a], graph.ids[coll[c]], 1.0))
    return out


if __name__ == "__main__":
    graph, res = SkillGraph(), load("dih.json")
    coll = collection(graph)
    with torch.no_grad():
        for space, feats in (("words", word_features), ("comention", comention_features)):
            t = time.time()
            F = feats(graph, coll)
            _, _, invcl = measures(F)
            built = time.time() - t
            rows = {m: report(graph, edges_topm(graph, coll, invcl, m), built) for m in MS}
            best = max(rows, key=lambda m: rows[m]["val A MAP"])
            res[f"invCL {space}"] = {**rows[best], "m": best, "per m": {m: {"A": r["A"]["MAP"], "B": r["B"]["MAP"], "val A": r["val A MAP"], "edges": r["edges"], "prec": r["precision vs closure"]} for m, r in rows.items()}}
            print(line(f"invCL {space}", res[f"invCL {space}"]), "m", best, file=sys.stderr)
            t = time.time()
            res[f"SLQS gen {space}"] = gen_report(graph, slqs_gen(graph, coll, F), time.time() - t)
            print(line(f"SLQS gen {space}", res[f"SLQS gen {space}"]), file=sys.stderr)
            save("dih.json", res)
