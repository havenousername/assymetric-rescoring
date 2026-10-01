"""LLM judges as rerankers: score p(A→B) for the cos + gen top 50 of every query, and both directions of every true
pair. Two judges from different model families, Claude Sonnet 5 (`claude -p`) and Codex (`codex exec`); Claude wrote
the labels, so a Claude judge alone would be circular. Raw answers are cached in data/llm_cache/ by prompt hash."""
import hashlib
import json
import random
import re
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor

import torch

from ..data import ROOT
from ..tasks import up_known, up_new
from .cosine import cosine_generality

CACHE = ROOT / "data" / "llm_cache"
K, PER_CALL, PAIRS_PER_CALL, WORKERS = 50, 5, 100, 6
DEF = ("p(A→B) = probability that a person with solid working experience in skill A has at least basic working "
       "competence in skill B, because working with A requires or exercises B. Judge real-world practice, "
       "not mere co-occurrence.")


def claude(prompt):
    return subprocess.run(["claude", "-p", "--model", "sonnet", "--setting-sources", "", "--tools", "",
                           "--strict-mcp-config", "--no-session-persistence",
                           "--system-prompt", "You are a precise annotator. Output only JSON."],
                          input=prompt, capture_output=True, text=True, timeout=900).stdout


def codex(prompt):
    with tempfile.NamedTemporaryFile("r", suffix=".txt") as f:
        subprocess.run(["codex", "exec", "--skip-git-repo-check", "-s", "read-only", "--ephemeral", "-o", f.name, "-"],
                       input=prompt, capture_output=True, text=True, timeout=900)
        return f.read()


JUDGES = {"claude-sonnet-5": claude, "codex-gpt-6-astra": codex}


def parse(out):
    m = re.search(r"\{.*\}", out or "", re.S)
    try: return json.loads(m.group(0)) if m else None
    except json.JSONDecodeError: return None


def ask(judge, prompt):
    """The judge's JSON answer, from the cache when this exact prompt was asked before."""
    path = CACHE / f"{judge}_{hashlib.sha1(prompt.encode()).hexdigest()[:16]}.txt"
    if path.exists(): return parse(path.read_text())
    for _ in range(3):
        out = JUDGES[judge](prompt)
        if parse(out) is not None:
            CACHE.mkdir(exist_ok=True)
            path.write_text(out)
            return parse(out)
    print(f"{judge}: unparseable after 3 tries", file=sys.stderr)


def describe(graph, i, wiki=0):
    n = graph.nodes[graph.ids[i]]
    return f"{n['label']} — {n['desc']}" + (f" ({n['wiki'][:wiki]})" if wiki and n["wiki"] else "")


def rank_prompt(graph, batch):
    lines = [DEF, "", "For each query skill A, score p(A→B) for every candidate B.",
             'Answer with one JSON object only: {"Q1": {"1": p, "2": p, ...}, "Q2": {...}}, p in [0, 1], two decimals.', ""]
    for n, (q, cands) in enumerate(batch, 1):
        lines += [f"Q{n}. A = {describe(graph, q, 300)}"] + [f"  {j}. {describe(graph, c)}" for j, c in enumerate(cands, 1)]
    return "\n".join(lines)


def pair_prompt(graph, batch):
    lines = [DEF, "", 'Score p(A→B) for every numbered pair. Answer with one JSON object only: {"1": p, "2": p, ...}, '
             "p in [0, 1], two decimals.", ""]
    return "\n".join(lines + [f"{j}. A = {describe(graph, a)} | B = {describe(graph, b)}" for j, (a, b) in enumerate(batch, 1)])


def test_tasks(graph):
    """The test queries of Tasks A and B with their cos + gen top K (after the exclusions evaluate applies), and the
    true pairs, for direction accuracy. The cos + gen top 50 holds 99% (A) / 93% (B) of the test answers."""
    base, catalogue = cosine_generality(graph.X, graph.generality), torch.tensor(graph.catalogue)
    tasks, true = [], set()
    for rel, excluded in [(up_new(graph, "test"), lambda q: set()),
                          (up_known(graph, "test"), lambda q: graph.ancestors_train[q] | {q})]:
        S = base(torch.tensor(list(rel)), catalogue).cpu()
        for q, row in zip(rel, S):
            order = [graph.catalogue[j] for j in row.argsort(descending=True).tolist()]
            tasks.append((q, [c for c in order if c not in excluded(q)][:K]))
        true |= {(q, b) for q in rel for b in rel[q]}
    return tasks, sorted(true)


def judge_all(graph, judge):
    """Ask the judge about every shortlist and both directions of every true pair.
    Returns (rank {A: {B: p}}, pair {(A, B): p}, "missing/asked")."""
    tasks, true = test_tasks(graph)
    both = true + [(b, a) for a, b in true]
    random.Random(0).shuffle(both)  # the two directions of a pair never sit next to each other
    rank_batches = [tasks[i:i + PER_CALL] for i in range(0, len(tasks), PER_CALL)]
    pair_batches = [both[i:i + PAIRS_PER_CALL] for i in range(0, len(both), PAIRS_PER_CALL)]
    with ThreadPoolExecutor(WORKERS) as ex:
        rank_answers = list(ex.map(lambda b: ask(judge, rank_prompt(graph, b)), rank_batches))
        pair_answers = list(ex.map(lambda b: ask(judge, pair_prompt(graph, b)), pair_batches))
    rank, pair, missing = {}, {}, 0
    for batch, answer in zip(rank_batches, rank_answers):
        for n, (q, cands) in enumerate(batch, 1):
            got = (answer or {}).get(f"Q{n}") or {}
            for j, c in enumerate(cands, 1):
                try: rank.setdefault(q, {})[c] = float(got[str(j)])
                except (KeyError, TypeError, ValueError): missing += 1
    for batch, answer in zip(pair_batches, pair_answers):
        for j, ab in enumerate(batch, 1):
            try: pair[ab] = float((answer or {})[str(j)])
            except (KeyError, TypeError, ValueError): missing += 1
    return rank, pair, f"{missing}/{sum(len(c) for _, c in tasks) + len(both)}"


class Judged:
    """Reranker: the judged top K first, ordered by cos + gen + p; everything else keeps cos + gen order.
    Ordering by p alone (cos + gen only breaking ties) scored lower on 3 of 4 test MAPs: Claude A 0.779 / B 0.681,
    Codex 0.726 / 0.630 vs 0.773 / 0.698 and 0.757 / 0.686 for the blend.
    rank: {A: {B: p}} for the shortlists; pair: {(A, B): p} for direction accuracy."""

    def __init__(self, graph, rank, pair):
        self.base, self.rank, self.pair = cosine_generality(graph.X, graph.generality), rank, pair

    def __call__(self, a, b):
        S, col = self.base(a, b).clone(), {c: j for j, c in enumerate(b.tolist())}
        for i, q in enumerate(a.tolist()):
            for c, p in self.rank.get(q, {}).items():
                if c in col: S[i, col[c]] += 10 + p  # cos + gen < 1.3, so every judged candidate outranks the rest
        return S

    def pairs(self, a, b):
        return torch.tensor([self.pair.get(ab, 0.0) for ab in zip(a.tolist(), b.tolist())])
