"""The skill graph: nodes, the catalogue, splits, ancestor sets, frozen text embeddings and the generality prior.

Every skill is a node index (position in the sorted Wikidata ids). A → B means "whoever has A has B".
The catalogue is the 925 skills that appear in a training edge; held-out skills (unseen at training time) and
hidden pairs (implications removed from the training graph) are the test and validation targets.
"""
import json
import math
from functools import cache
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
ENCODER = "BAAI/bge-base-en-v1.5"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


@cache
def sentence_encoder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(ENCODER, device=DEVICE)


def encode(texts):
    """Texts → unit-length bge-base vectors [len(texts), 768] on DEVICE."""
    vectors = sentence_encoder().encode(texts, normalize_embeddings=True, batch_size=64)
    return torch.tensor(vectors, dtype=torch.float32, device=DEVICE)


class SkillGraph:
    def __init__(self, path=ROOT / "data" / "graph.json"):
        g = json.loads(Path(path).read_text())
        test = {tuple(p) for p in g["new_pairs"]["test"]}  # pairs implied by both val and test edges stay in test only
        g["new_pairs"]["val"] = [p for p in g["new_pairs"]["val"] if tuple(p) not in test]

        self.nodes = g["nodes"]
        self.ids = sorted(self.nodes)
        self.index = {q: i for i, q in enumerate(self.ids)}
        self.node_ids = torch.arange(len(self.ids), device=DEVICE)  # inputs for free-vector heads
        ix = self.index

        self.catalogue = sorted({ix[q] for e in g["train_edges"] for q in e})
        catalogue_qids = {self.ids[i] for i in self.catalogue}
        self.ancestors_train = {ix[a]: {ix[b] for b in bs} for a, bs in g["train_ancestors"].items() if a in catalogue_qids}
        self.ancestors = {ix[a]: {ix[b] for b in bs} for a, bs in g["full_ancestors"].items()}
        self.descendants_train = {b: [a for a in self.ancestors_train if b in self.ancestors_train[a]] for b in range(len(self.ids))}
        self.positives = [(a, b) for a, bs in self.ancestors_train.items() for b in bs]  # training closure

        self.heldout = {split: [ix[q] for q in qs] for split, qs in g["heldout_nodes"].items()}
        self.hidden_pairs = {split: [(ix[a], ix[b]) for a, b in ps] for split, ps in g["new_pairs"].items()}

        self.X = self._embeddings()
        n_desc = [0] * len(self.ids)
        for bs in self.ancestors_train.values():
            for b in bs: n_desc[b] += 1
        self.generality = torch.tensor([math.log1p(n) for n in n_desc], device=DEVICE)  # log(1 + #train descendants)

    def label(self, i):
        return self.nodes[self.ids[i]]["label"]

    def text(self, i):
        """What the encoder reads for a skill."""
        n = self.nodes[self.ids[i]]
        return f"{n['label']}. {n['desc']}. {n['wiki'][:600]}".strip()

    def by_label(self, label):
        return next(i for i, q in enumerate(self.ids) if self.nodes[q]["label"] == label)

    def add_texts(self, texts):
        """Encode new texts (a CV, a free-text query) as extra rows of X, with generality 0 (no known descendants),
        so index-level scorers score them like any node. In place, so scorers built earlier see the new rows."""
        new = encode(texts)
        self.X.data = torch.cat([self.X, new])
        self.generality.data = torch.cat([self.generality, self.generality.new_zeros(len(texts))])
        return list(range(len(self.X) - len(texts), len(self.X)))

    def _embeddings(self):
        path = ROOT / "data" / f"emb_{ENCODER.replace('/', '_')}.npy"
        if not path.exists():
            np.save(path, encode([self.text(i) for i in range(len(self.ids))]).cpu().numpy())
        return torch.tensor(np.load(path), dtype=torch.float32, device=DEVICE)
