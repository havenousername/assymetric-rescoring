"""Local open-weight judges on the 2,630 candidate pairs Claude labelled y/n (data/labels, pipeline/build_graph.py):
does a model that runs on this machine agree with Claude? Each judge scores p(A→B) and p(B→A):
- NLI cross-encoder: entailment probability of a hypothesis template given a premise.
- Instruct LLM (MLX, 4-bit): softmax over the Yes and No logits of the first answer token, no generation.
Baselines: cosine of the two texts (no direction) and the number of candidate sources (the LLM-free rule).
Usage: uv run --with mlx-lm python -m experiments.local_judge [judge ...] → results/local_judge.json
(+ per-pair p(A→B), p(B→A) in results/local_judge_scores.json, for experiments/nollm_gen.py)"""
import csv
import glob
import json
import math
import sys
import time

import numpy as np
import torch
from scipy.stats import rankdata

from skillmatch.data import DEVICE, encode
from skillmatch.methods.llm_judge import DEF

from .zoo import load, save

NLIS = {"nli": "MoritzLaurer/deberta-v3-large-zeroshot-v2.0", "nli-base": "MoritzLaurer/deberta-v3-base-zeroshot-v2.0",
        "nli-modernbert": "MoritzLaurer/ModernBERT-base-zeroshot-v2.0"}
NLI_TEMPLATES = {  # (premise, hypothesis); a/b = label (description), A/B = label
    "person": ("This person has solid working experience with {a}.", "This person has at least basic working competence in {B}."),
    "relation": ("{a}. {b}.", "Working with {A} requires or exercises {B}."),
}
LLMS = {"qwen2.5-3b": "mlx-community/Qwen2.5-3B-Instruct-4bit", "gemma3-12b": "mlx-community/gemma-3-12b-it-4bit"}
QUESTION = "{d}\n\nA: {a}\nB: {b}\n\nDoes a person with solid working experience in A have at least basic working competence in B? Answer Yes or No."


def pairs():
    """[(A, B, sources, y)] in candidate id order (data/candidates.json), A, B the "label — description" text Claude saw."""
    out = {}
    for f in sorted(glob.glob("data/label_batches/batch_*.txt")):
        shown = {}
        for line in open(f):
            i, a, b, evidence = line.rstrip("\n").split(" | ")
            shown[i] = a.removeprefix("A: "), b.removeprefix("B: "), {e.split(":")[0] if e.startswith("so") else e for e in evidence.split(", ")}
        out |= {int(r["id"]): (*shown[r["id"]], r["y"] == "y") for r in csv.DictReader(open(f.replace("label_batches", "labels").replace(".txt", ".csv")))}
    assert sorted(out) == list(range(len(out)))
    return [out[i] for i in range(len(out))]


def split(text):
    label, _, desc = text.partition(" — ")
    return label, f"{label} ({desc.strip()})" if desc.strip() else label


def both(score, P):
    """score([(x, y)]) → p(x→y) per pair, run on A→B and on B→A."""
    return np.array(score([(a, b) for a, b, *_ in P])), np.array(score([(b, a) for a, b, *_ in P]))


def cosine(P):
    texts = sorted({t for a, b, *_ in P for t in (a, b)})
    X, row = encode(texts), {t: i for i, t in enumerate(texts)}
    s = np.array([(X[row[a]] @ X[row[b]]).item() for a, b, *_ in P])
    return s, None, {}


def sources(P):
    return np.array([len(s) for *_, s, _ in P], dtype=float), None, {}


def nli_scorer(repo, template):
    """score([(x, y)]) → entailment probability of the template, x and y "label — description" texts."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok, model = AutoTokenizer.from_pretrained(repo), AutoModelForSequenceClassification.from_pretrained(repo).to(DEVICE).eval()
    entail, (premise, hypothesis) = model.config.label2id["entailment"], NLI_TEMPLATES[template]

    @torch.no_grad()
    def score(xy, batch=32):
        out = []
        for i in range(0, len(xy), batch):
            (A, a), (B, b) = zip(*[split(x) for x, _ in xy[i:i + batch]]), zip(*[split(y) for _, y in xy[i:i + batch]])
            enc = tok([premise.format(A=p, a=q, B=r, b=s) for p, q, r, s in zip(A, a, B, b)],
                      [hypothesis.format(A=p, a=q, B=r, b=s) for p, q, r, s in zip(A, a, B, b)],
                      padding=True, truncation=True, return_tensors="pt").to(DEVICE)
            out += model(**enc).logits.softmax(-1)[:, entail].tolist()
        return out
    return score


def nli(repo, template):
    return lambda P: (*both(nli_scorer(repo, template), P), {})


def llm(repo):
    def judge(P):
        import mlx.core as mx
        from mlx_lm import load as load_mlx
        model, tok = load_mlx(repo)
        yes, no = (tok.encode(w, add_special_tokens=False) for w in ("Yes", "No"))
        assert len(yes) == len(no) == 1, (yes, no)
        mass = []

        def score(xy):
            out, t = [], time.time()
            for i, (x, y) in enumerate(xy):
                ids = tok.apply_chat_template([{"role": "user", "content": QUESTION.format(d=DEF, a=x, b=y)}], add_generation_prompt=True)
                logits = model(mx.array([ids]))[0, -1]
                lp = logits - mx.logsumexp(logits)
                ly, ln = lp[yes[0]].item(), lp[no[0]].item()
                out.append(1 / (1 + math.exp(ln - ly)))
                mass.append(math.exp(ly) + math.exp(ln))
                if i % 250 == 0: print(f"  {repo} {i}/{len(xy)} {(time.time() - t) / (i + 1) * 1000:.0f} ms/pair", file=sys.stderr, flush=True)
            return out
        return *both(score, P), {"yes/no mass": float(np.mean(mass))}
    return judge


JUDGES = {"cosine": cosine, "sources": sources, **{f"{m}-{t}": nli(r, t) for m, r in NLIS.items() for t in NLI_TEMPLATES}, **{k: llm(v) for k, v in LLMS.items()}}


def auc(y, s):
    """P(a random y pair outscores a random n pair), ties half (Mann-Whitney)."""
    r, pos = rankdata(s), y.sum()
    return (r[y].sum() - pos * (pos + 1) / 2) / (pos * (len(y) - pos))


def report(P, y, fwd, rev, seconds):
    groups = {"SO only": [s == {"so"} for *_, s, _ in P], "Wikidata only": ["so" not in s for *_, s, _ in P],
              "SO + Wikidata": ["so" in s and len(s) > 1 for *_, s, _ in P]}
    r = {"AUC": auc(y, fwd), **{f"AUC {g}": auc(y[np.array(m)], fwd[np.array(m)]) for g, m in groups.items()}, "s": seconds}
    if rev is not None:
        r |= {"AUC fwd−rev": auc(y, fwd - rev), "dir (y pairs: fwd > rev)": np.mean((fwd[y] > rev[y]) + 0.5 * (fwd[y] == rev[y])),
              "precision@0.5": y[fwd >= 0.5].mean(), "recall@0.5": (fwd[y] >= 0.5).mean(), "ms/pair": 1000 * seconds / (2 * len(y))}
    return r


if __name__ == "__main__":
    assert auc(np.array([False, True, True]), np.array([0.1, 0.9, 0.1])) == 0.75
    P, res, scores = pairs(), load("local_judge.json"), load("local_judge_scores.json")
    y = np.array([p[3] for p in P])
    print(len(P), "pairs,", y.mean().round(3), "labelled y", file=sys.stderr)
    for name in sys.argv[1:] or JUDGES:
        t = time.time()
        fwd, rev, extra = JUDGES[name](P)
        res[name] = {**report(P, y, fwd, rev, time.time() - t), **extra}
        save("local_judge.json", res)
        if rev is not None: scores[name] = {"fwd": fwd.tolist(), "rev": rev.tolist()}; save("local_judge_scores.json", scores)
        print(name, json.dumps(res[name], default=float), file=sys.stderr, flush=True)
