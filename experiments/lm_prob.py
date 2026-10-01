"""Directional token probabilities of a small causal LM as an implication judge, no embedding: Qwen/Qwen2.5-1.5B
(base, bf16, 3.1 GB download, HF org Qwen) scores log P(" B" | template(A)) teacher-forced, both directions, with
Hearst / prerequisite templates. Measures: raw sum log-prob, its mean per token, and PMI = log P(B | template(A))
− log P(B | template(generic A)), which cancels B's own prior.
pilot: AUC against Claude's y on the labelled candidate pairs the evaluation never touches (no held-out node, no
hidden pair), per template and measure, plus direction accuracy (fwd > rev on y pairs).
judge <set>: score a hub_nli candidate set with the pilot's best template; A → B kept when the directional score
favours A → B and the pair is in the top-q share by score (q picked on val A MAP), cos + gen as everywhere else.
Usage: uv run python -m experiments.lm_prob pilot | judge <set ...> → results/lm_prob.json (scores cached in
results/lm_prob_scores.json)."""
import sys
import time

import numpy as np
import torch

from skillmatch.data import DEVICE, SkillGraph

from .hub_nli import SETS
from .local_graph import line, report
from .local_judge import auc, pairs
from .zoo import load, save

REPO = "Qwen/Qwen2.5-1.5B"
TEMPLATES = {"kind": "{A} is a kind of", "requires": "Using {A} requires knowing", "built": "{A} is built on top of",
             "knows": "Anyone who knows {A} also knows", "prereq": "Prerequisite skill for {A}:"}
GENERIC = "this technology"
SHARES = (0.1, 0.2, 0.3, 0.5, 1.0)


class LM:
    def __init__(self, repo=REPO):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(repo)
        self.tok.padding_side = "right"
        self.model = AutoModelForCausalLM.from_pretrained(repo, dtype=torch.bfloat16).to(DEVICE).eval()

    @torch.no_grad()
    def logprob(self, prefixes, conts, batch=48):
        """(sum, mean) log P(cont tokens | prefix) per pair."""
        sums, means = [], []
        for i in range(0, len(prefixes), batch):
            P, C = prefixes[i:i + batch], conts[i:i + batch]
            n_pre = [len(self.tok(p).input_ids) for p in P]
            enc = self.tok([p + " " + c for p, c in zip(P, C)], return_tensors="pt", padding=True).to(DEVICE)
            lp = self.model(**enc).logits[:, :-1].float().log_softmax(-1)
            tgt = enc.input_ids[:, 1:]
            tok_lp = lp.gather(-1, tgt[..., None])[..., 0]
            mask = enc.attention_mask[:, 1:].clone()
            for r, n in enumerate(n_pre): mask[r, :n - 1] = 0
            s = (tok_lp * mask).sum(1)
            sums += s.tolist(); means += (s / mask.sum(1).clamp(min=1)).tolist()
        return np.array(sums), np.array(means)


def score_pairs(lm, ab, template):
    """{"raw", "mean", "pmi"} arrays for label pairs [(A, B)], one direction."""
    s, m = lm.logprob([template.format(A=a) for a, _ in ab], [b for _, b in ab])
    g, _ = lm.logprob([template.format(A=GENERIC)] * len(ab), [b for _, b in ab])
    return {"raw": s, "mean": m, "pmi": s - g}


def pilot_pairs(graph):
    """(ia, ib, y) for labelled candidate pairs with both skills in the collection and no hidden pair either way."""
    by_label, held = {}, {i for s in graph.heldout.values() for i in s}
    for i, q in enumerate(graph.ids): by_label.setdefault(graph.nodes[q]["label"], i)
    hidden = {p for ps in graph.hidden_pairs.values() for p in ps}
    out = []
    for a, b, _, y in pairs():
        ia, ib = by_label.get(a.split(" — ")[0]), by_label.get(b.split(" — ")[0])
        if ia is None or ib is None or {ia, ib} & held or (ia, ib) in hidden or (ib, ia) in hidden: continue
        out.append((ia, ib, y))
    return out


def pilot(graph, lm):
    P = pilot_pairs(graph)
    y = np.array([p[2] for p in P])
    ab = [(graph.label(a), graph.label(b)) for a, b, _ in P]
    res = {"pairs": len(P), "y rate": float(y.mean())}
    for name, tpl in TEMPLATES.items():
        t = time.time()
        f, r = score_pairs(lm, ab, tpl), score_pairs(lm, [(b, a) for a, b in ab], tpl)
        sec = time.time() - t
        for k in f:
            d = f[k] - r[k]
            res[f"{name}/{k}"] = {"AUC": auc(y, f[k]), "AUC fwd−rev": auc(y, d), "dir": float(np.mean((d[y] > 0) + 0.5 * (d[y] == 0))), "ms/pair": 1000 * sec / (2 * len(P))}
        print(name, {k: {m: round(v, 3) for m, v in res[f'{name}/{k}'].items()} for k in f}, file=sys.stderr, flush=True)
    return res


def judged(graph, lm, P, template, measure, key):
    cache = load("lm_prob_scores.json")
    C = cache.setdefault(key, {})
    todo = sorted(p for p in P if f"{p[0]},{p[1]}" not in C)
    t = time.time()
    if todo:
        ab = [(graph.label(i), graph.label(j)) for i, j in todo]
        f, r = score_pairs(lm, ab, template)[measure], score_pairs(lm, [(b, a) for a, b in ab], template)[measure]
        C.update({f"{i},{j}": [float(x), float(z)] for (i, j), x, z in zip(todo, f, r)})
        save("lm_prob_scores.json", cache)
    return {p: tuple(C[f"{p[0]},{p[1]}"]) for p in P}, time.time() - t, len(todo)


def edges(graph, S, share):
    trip = [((i, j, f) if f > b else (j, i, b)) for (i, j), (f, b) in S.items()]
    cut = np.quantile([p for _, _, p in trip], 1 - share) if share < 1 else -np.inf
    return [(graph.ids[a], graph.ids[c], 1.0) for a, c, p in trip if p >= cut]


def judge(graph, lm, name, template, measure):
    t0 = time.time()
    P = SETS[name](graph)
    S, sec, new = judged(graph, lm, P, TEMPLATES[template], measure, f"{template}/{measure}")
    build = time.time() - t0
    rows = {q: report(graph, edges(graph, S, q), build) for q in SHARES}
    best = max(rows, key=lambda q: rows[q]["val A MAP"])
    return {**rows[best], "share": best, "pairs": len(P), "LM s (new pairs)": sec, "new pairs": new, "template": template, "measure": measure,
            "per share": {q: {"A": r["A"]["MAP"], "B": r["B"]["MAP"], "val A": r["val A MAP"], "edges": r["edges"], "prec": r["precision vs closure"]} for q, r in rows.items()}}


if __name__ == "__main__":
    graph, res = SkillGraph(), load("lm_prob.json")
    lm = LM()
    with torch.no_grad():
        if sys.argv[1] == "pilot":
            res["pilot"] = pilot(graph, lm)
        else:
            best = max((k for k in res["pilot"] if "/" in k), key=lambda k: res["pilot"][k]["AUC fwd−rev"])
            template, measure = best.split("/")
            for name in sys.argv[2:]:
                res[f"LM {name}"] = judge(graph, lm, name, template, measure)
                print(line(f"LM {name}", res[f"LM {name}"]), f"| share {res[f'LM {name}']['share']} {best}", file=sys.stderr)
        save("lm_prob.json", res)
