"""Keyword implication graph from co-occurrence across CVs: no labels, no taxonomy, no LLM.

Subsumption (Sanderson & Croft 1999), adapted to CVs: A → B ("whoever has A has B") when
- P(B | A) ≥ tau: most CVs naming A also name B,
- n(B) ≥ min_ratio · n(A): B is clearly more common, so near-equivalents (HTML, CSS) get no edge and the graph is acyclic,
- P(B | A) / P(B) ≥ min_lift: the overlap is well above B's base rate, so hubs named in a third or more of all CVs
  (Developer, Git) can't be a target and don't absorb everything,
- n(A, B) ≥ min_co: enough CVs to trust the ratio,
- B is not part of A's own name (Unreal Engine, Engine): such pairs co-occur by construction, so they say nothing.
Optionally a judge (e.g. a local NLI model, see nli_judge) keeps only the edges it reads in the same direction.

KeywordGraph exposes what skillmatch's trainers and hybrid() read: X, catalogue, positives (the transitive closure),
ancestors_train, descendants_train, generality = log(1 + #descendants).
"""
import math
from dataclasses import dataclass

import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch

from skillmatch.data import DEVICE


@dataclass
class Cooccurrence:
    n: np.ndarray       # [V] CVs naming each keyword
    pairs: sp.csr_matrix  # [V, V] CVs naming both
    cvs: int

    @classmethod
    def count(cls, keyword_sets, vocab_size):
        rows = np.repeat(np.arange(len(keyword_sets)), [len(s) for s in keyword_sets])
        cols = np.fromiter((k for s in keyword_sets for k in s), dtype=np.int64, count=len(rows))
        A = sp.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(len(keyword_sets), vocab_size))
        return cls(np.asarray(A.sum(0)).ravel(), (A.T @ A).tocsr(), len(keyword_sets))


def subsumption_edges(co, contained=frozenset(), tau=0.5, min_ratio=1.5, min_lift=3.0, min_co=5):
    """[(a, b, P(b|a))] for every pair that passes the rule in the module docstring. contained: Vocabulary.contained()."""
    P = co.pairs.tocoo()
    a, b, nab = P.row, P.col, P.data
    na, nb = co.n[a], co.n[b]
    p = nab / na
    keep = (a != b) & (nab >= min_co) & (p >= tau) & (nb >= min_ratio * na) & (p / (nb / co.cvs) >= min_lift)
    return [(int(x), int(y), float(z)) for x, y, z in zip(a[keep], b[keep], p[keep]) if (x, y) not in contained]


class KeywordGraph:
    def __init__(self, vocab, edges, X):
        self.vocab, self.X = vocab, X
        self.G = nx.DiGraph()
        self.G.add_nodes_from(range(len(vocab)))
        self.G.add_weighted_edges_from(edges)
        assert nx.is_directed_acyclic_graph(self.G)
        self.catalogue = list(range(len(vocab)))
        self.ancestors_train = {a: nx.descendants(self.G, a) for a in self.G}  # edges point specific → general
        self.descendants_train = {b: [] for b in self.G}
        for a, bs in self.ancestors_train.items():
            for b in bs: self.descendants_train[b].append(a)
        self.positives = [(a, b) for a, bs in self.ancestors_train.items() for b in bs]
        self.generality = torch.tensor([math.log1p(len(self.descendants_train[b])) for b in self.G], device=DEVICE)

    def implies(self, name):
        """Direct edges out of a keyword: what having it implies, most likely first."""
        a = self.vocab.find(name)
        return sorted(((self.vocab[b], round(d["weight"], 2)) for b, d in self.G[a].items()), key=lambda x: -x[1])

    def implied_by(self, name):
        b = self.vocab.find(name)
        return sorted(((self.vocab[a], round(self.G[a][b]["weight"], 2)) for a in self.G.predecessors(b)), key=lambda x: -x[1])

    def stats(self):
        nodes = sum(1 for v in self.G if self.G.degree(v))
        return {"keywords": len(self.vocab), "keywords with an edge": nodes, "edges": self.G.number_of_edges(),
                "closure pairs": len(self.positives), "longest chain": nx.dag_longest_path_length(self.G, weight=None)}


def nli_judge(model="MoritzLaurer/deberta-v3-large-zeroshot-v2.0", batch=64):
    """A local NLI cross-encoder as a judge: (names_a, names_b) → P(entailment) that having A means having B."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model)
    net = AutoModelForSequenceClassification.from_pretrained(model).to(DEVICE).eval()
    entail = net.config.label2id["entailment"]

    @torch.no_grad()
    def judge(a_names, b_names):
        out = []
        for i in range(0, len(a_names), batch):
            prem = [f"This candidate has hands-on experience with {a}." for a in a_names[i:i + batch]]
            hyp = [f"This candidate has experience with {b}." for b in b_names[i:i + batch]]
            enc = tok(prem, hyp, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
            out += net(**enc).logits.softmax(-1)[:, entail].tolist()
        return np.array(out)
    return judge


def verify(edges, vocab, judge, t=0.5):
    """Keep the edges the judge accepts in their own direction: p(a→b) ≥ t and p(a→b) > p(b→a)."""
    a = [vocab[x] for x, _, _ in edges]
    b = [vocab[y] for _, y, _ in edges]
    fwd, rev = judge(a, b), judge(b, a)
    return [e for e, f, r in zip(edges, fwd, rev) if f >= t and f > r]


def build_graph(cvs, X=None, judge=None, **rule):
    """The whole construction in one call. cvs: [(body, text)], body = the CV's free text minus its title (mined for
    keywords), text = everything (searched for them). X: keyword vectors, default bge-base on the names. judge: an
    optional verifier (nli_judge()). rule: subsumption_edges thresholds. Returns (graph, keyword sets per CV)."""
    from .data import Vocabulary
    from .models import embed_keywords
    vocab = Vocabulary.mine(body for body, _ in cvs)
    sets = [vocab.extract(text) for _, text in cvs]
    edges = subsumption_edges(Cooccurrence.count(sets, len(vocab)), vocab.contained(), **rule)
    if judge is not None: edges = verify(edges, vocab, judge)
    return KeywordGraph(vocab, edges, embed_keywords(vocab) if X is None else X), sets
