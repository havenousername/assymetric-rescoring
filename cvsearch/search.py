"""Search CVs with a keyword scorer, and the evaluation grid.

A CV is the set of keywords it names. For a requirement r:
- down ("Looking for C++ experts"): CV score = max over its keywords k of s(k → r), who has r;
- up ("would be nice if has LLVM"):  CV score = max over its keywords k of s(r → k), what r implies, as a fallback when
  few CVs name r itself, plus the down score: among CVs with C++, the ones that look most like LLVM people come first
  (Clang, GCC), not every CV that mentions C++ once.
The mean of the CV's three best keywords adds TIE times its value, so more support counts: a CV with
C++ and Clang beats one with C++ alone for "LLVM". Search lists CVs that name r first, whatever the
score: that is the identity correspondence, a CV that says LLVM answers "LLVM" before any CV that only implies it.
A request's keywords are read longest match first; keywords a third or more of the indexed CVs name ("Developer") match
too many CVs to be a requirement and are dropped, the same bound that keeps them from being graph targets.

Evaluation, on the Primary Keyword p the candidate picked (TECH), never used to build anything:
- down seen / unseen: query p's keyword; rank the CVs that name p / that don't; relevant = Primary Keyword p. MAP.
- up seen / unseen:   for each CV, rank the TECH keywords by its score toward each; relevant = its own p, when the
                      CV names p / doesn't. MRR and accuracy@1, averaged over the Primary Keywords.
- identity:           keyword level, share of keywords that come first in their own down list; CV level, AUC of
                      CVs naming p over CVs that don't, on the raw score.
Keyword search (does the CV name it?) is the baseline: perfect on "names p", blind on the rest.
"""
import math
from collections import Counter

import numpy as np
import torch

from .data import TECH

ALIASES = {"JavaScript": ["js"], "Golang": ["go"], "C++": ["cpp"], "Node.js": ["node"]}  # evaluation only: what counts as naming p


COMMON = 1 / 3  # share of indexed CVs above which a keyword is too common to be a requirement
TIE = 0.1  # weight of the mean of a CV's three best keywords next to its best one


class CVIndex:
    def __init__(self, cvs, vocab):
        self.cvs, self.vocab = cvs, vocab
        self.keywords = [vocab.extract(c["text"]) for c in cvs]
        self.sets = [set(k) for k in self.keywords]
        self.share = {k: n / len(cvs) for k, n in Counter(k for s in self.sets for k in s).items()}
        self.idx = torch.full((len(cvs), max(map(len, self.keywords), default=1) or 1), -1, dtype=torch.long)
        for i, k in enumerate(self.keywords): self.idx[i, :len(k)] = torch.tensor(k, dtype=torch.long)

    def __len__(self):
        return len(self.cvs)

    def names(self, ids):
        """[N] bool: the CV names one of the keyword ids."""
        return torch.tensor([bool(s & ids) for s in self.sets])

    def score(self, col):
        """Keyword scores [V] → per CV (score, the keyword that gave it)."""
        col = col.float().cpu()
        v = torch.where(self.idx >= 0, col[self.idx.clamp(min=0)], torch.tensor(-math.inf))
        best, arg = v.max(1)
        top = v.topk(min(3, v.shape[1]), dim=1).values
        tie = torch.where(torch.isfinite(top), top, best[:, None]).mean(1)
        return best + TIE * torch.nan_to_num(tie, neginf=0.0), self.idx.gather(1, arg[:, None])[:, 0]


@torch.no_grad()
def column(M, r, mode, V):
    """s(k → r) for every keyword k (down), or s(r → k) (up)."""
    allk, q = torch.arange(V), torch.tensor([r])
    return (M(allk, q)[:, 0] if mode == "down" else M(q, allk)[0]).float().cpu()


def cv_scores(index, M, r, mode):
    """Per CV: (score, the keyword that gave it, the keyword that ties it to r). Down: who has r. Up: what r implies,
    plus who has r, so the closest of the CVs with what r implies come first."""
    V = len(index.vocab)
    down, close = index.score(column(M, r, "down", V))
    if mode == "down": return down, close, close
    up, via = index.score(column(M, r, "up", V))
    return up + down, via, close


class Search:
    def __init__(self, index, scorers):
        self.index, self.scorers = index, scorers

    def requirements(self, text):
        return [r for r in self.index.vocab.requirements(text) if self.index.share.get(r, 0) < COMMON]

    def query(self, text, mode="down", method="graph", k=10, names=None):
        """Ranked CVs for a free-text request: (requirement keywords, rows). The requirements are the vocabulary
        keywords the text names. names=True / False keeps only CVs that name / don't name a requirement."""
        vocab, req = self.index.vocab, self.requirements(text)
        if not req: return req, []
        total, named, because, close = torch.zeros(len(self.index)), torch.zeros(len(self.index)), [], []
        for r in req:
            s, b, c = cv_scores(self.index, self.scorers[method], r, mode)
            total += s
            named += self.index.names({r}).float()
            because.append(b); close.append(c)
        order = sorted((i for i in range(len(self.index)) if names is None or bool(named[i]) == names),
                       key=lambda i: (-named[i].item(), -total[i].item()))[:k]
        name = lambda t, i: vocab[t[i].item()] if t[i] >= 0 else None
        rows = [{"cv": self.index.cvs[i], "names": int(named[i]), "score": total[i].item(),
                 "because": [name(b, i) for b in because], "close": [name(c, i) for c in close]} for i in order]
        return [vocab[r] for r in req], rows


def _ap(scores, relevant, seed=0):
    """Average precision, ties broken at random (seeded)."""
    if not relevant.any(): return math.nan
    perm = torch.randperm(len(scores), generator=torch.Generator().manual_seed(seed))
    order = perm[torch.argsort(scores[perm], descending=True, stable=True)]
    ranks = relevant[order].nonzero()[:, 0].double() + 1
    return float((torch.arange(1, len(ranks) + 1) / ranks).mean())


def _auc(pos, neg):
    if not len(pos) or not len(neg): return math.nan
    allv = torch.cat([pos, neg])
    ranks = torch.empty(len(allv), dtype=torch.double)
    ranks[allv.argsort()] = torch.arange(1, len(allv) + 1, dtype=torch.double)
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def targets(vocab):
    """{p: (query keyword id, ids that count as naming p)}."""
    out = {}
    for p in TECH:
        q = vocab.find(p)
        out[p] = (q, {q} | {vocab.index[a] for a in ALIASES.get(p, []) if a in vocab.index})
    return out


def grid(index, M=None, identity=True, sample=1000, seed=0):
    """The evaluation grid for scorer M, or keyword search when M is None."""
    vocab, T = index.vocab, targets(index.vocab)
    pk = np.array([c["pk"] for c in index.cvs])
    named = {p: index.names(ids) for p, (_, ids) in T.items()}
    S = torch.stack([named[p].double() if M is None else index.score(column(M, q, "down", len(vocab)))[0].double()
                     for p, (q, _) in T.items()], 1)  # [N, P]: CV i toward p
    res = {}
    for kind, pick in (("seen", lambda m: m), ("unseen", lambda m: ~m)):
        aps = [_ap(S[pick(named[p]), j], torch.from_numpy(pk == p)[pick(named[p])]) for j, p in enumerate(T)]
        res[f"down {kind} MAP"] = float(np.nanmean(aps))
        rr, acc = [], []
        for j, p in enumerate(T):
            rows = np.where((pk == p) & pick(named[p]).numpy())[0]
            if not len(rows): continue
            onehot = torch.zeros(len(T), dtype=torch.bool); onehot[j] = True
            r = [_ap(S[i], onehot, seed=int(i)) for i in rows]
            rr.append(np.mean(r)); acc.append(np.mean([x == 1 for x in r]))
        res[f"up {kind} MRR"], res[f"up {kind} acc@1"] = float(np.mean(rr)), float(np.mean(acc))
    if identity and M is not None:
        res["identity: CV AUC"] = float(np.nanmean([_auc(S[named[p], j], S[~named[p], j]) for j, p in enumerate(T)]))
        rng = np.random.default_rng(seed)
        common = [i for i in range(len(vocab)) if vocab.df[i] >= 20]
        qs = list(rng.choice(common, min(sample, len(common)), replace=False))
        top = [column(M, int(r), "down", len(vocab)) for r in qs]
        res["identity: keyword top 1"] = float(np.mean([(c[int(r)] >= c.max()).item() for r, c in zip(qs, top)]))
    return res


UP_CHECK = [("Spring", "Java"), ("Spring Boot", "Java"), ("Hibernate", "Java"), ("LLVM", "C++"), ("Qt", "C++"),
            ("STL", "C++"), ("Unreal Engine", "C++"), ("Django", "Python"), ("Flask", "Python"), ("FastAPI", "Python"),
            ("Laravel", "PHP"), ("Symfony", "PHP"), ("SwiftUI", "iOS"), ("UIKit", "iOS"), ("Jetpack Compose", "Android"),
            ("Ruby on Rails", "Ruby"), ("Gin", "Golang"), ("ASP.NET", ".NET"), ("Entity Framework", ".NET"),
            ("NestJS", "Node.js"), ("Terraform", "DevOps"), ("Ansible", "DevOps")]  # hand-picked: the obvious fallback people


def up_check(search, method, k=20):
    """{request: share of its first k up-mode fallback CVs (not naming it) whose Primary Keyword is the obvious one}."""
    out = {}
    for r, p in UP_CHECK:
        req, rows = search.query(r, "up", method, k=k, names=False)
        if len(req) == 1 and rows: out[r] = float(np.mean([x["cv"]["pk"] == p for x in rows]))
    return out


def tune(index, make, options, objective=("down unseen MAP", "up unseen MRR")):
    """The option whose scorer make(option) has the best mean of the objective metrics on this (val) index."""
    scored = {o: grid(index, make(o), identity=False) for o in options}
    best = max(options, key=lambda o: np.mean([scored[o][m] for m in objective]))
    return best, scored
