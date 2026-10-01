"""Builds cvsearch.ipynb: CV search with implied skills, built from the CVs alone (package cvsearch/).
Usage: python3 tools/build_cvsearch.py cvsearch.ipynb, then from the repo root
       uv run --with nbconvert jupyter nbconvert --to notebook --execute --inplace cvsearch.ipynb"""
import json
import sys

out = sys.argv[1]
cells = []
def md(s): cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip()})
def code(s): cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip()})

md('''
# CV search with implied skills

A recruiter types "Looking for the C++ experts". Keyword search returns the CVs that say C++. It misses the people who
wrote "Qt, STL, LLVM, embedded Linux" and never typed the two characters, and it can't tell that a request for LLVM
experience is close to a request for C++. This notebook builds a search that can, from a collection of CVs alone:

1. **CVs**: 210k real, anonymized IT CVs from Djinni ([dataset](https://huggingface.co/datasets/lang-uk/recruitment-dataset-candidate-profiles-english), MIT; [paper](https://unlp.org.ua/wp-content/uploads/2024/06/drushchak-romanyshyn.pdf)). Skills sit inside the text as written.
2. **Keywords**: mined from the CVs' own skill lists. No outside vocabulary.
3. **Implication graph**: "whoever names A also names B" from co-occurrence across CVs. No labels, no taxonomy, no LLM.
4. **Scorers**: cosine, the graph itself, and order / box / hyperbolic embeddings trained on the graph.
5. **Evaluation**: on a ground truth the CVs carry but nothing above reads, the *Primary Keyword* each candidate picked on the platform.
6. **Search** with an up / down switch, the CVs that name the requirement first: the HR demo, also exported as a standalone page (section 8).

Nothing here reads the earlier skill graph (`data/graph.json`), its labels or its models. Only the method code is shared: the heads and trainers in `skillmatch/methods` and `skillmatch/training.py`.

Run from the repo root. The first run downloads the CVs (226 MB parquet) to `data/djinni/`. The whole notebook takes about 15 minutes on a laptop, most of it training the three embedding models. One seed (0); MPS training isn't bit-reproducible, so a rerun can move the trained rows a little. Every claim in the text cells is computed in the same run.
''')
code(r'''
import html
import os
import re
import time
import warnings

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
warnings.filterwarnings("ignore")

import numpy as np
import torch
from IPython.display import HTML, Markdown, display

from cvsearch import (HEADS, NAMES, TECH, CVIndex, Cooccurrence, KeywordGraph, Search, Vocabulary, column, embed_keywords,
                      eval_cvs, graph_cvs, grid, scorers, subsumption_edges, targets, train, tune)
from cvsearch.demo import scrub  # some CVs still carry profile URLs and phone numbers
from skillmatch.methods import cosine_generality, hybrid

def table(header, rows):
    """A markdown table."""
    fmt = lambda x: f"{x:.3f}" if isinstance(x, float) else str(x)
    return "\n".join(["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] +
                     ["| " + " | ".join(fmt(x) for x in r) + " |" for r in rows])
t0 = time.time()
''')

md('''
## 1. The CVs

Each CV is a candidate profile: a title (Position), free text about their experience (Moreinfo) and highlights. A fourth field, "Looking For", holds wishes like "friendly team", so it's left out.

Splits are by a hash of the CV id:
- **graph split (80%)**: the vocabulary and the implication graph are built from these CVs, whatever their Primary Keyword.
- **val (5%) and test (15%)**: CVs whose Primary Keyword is one of 18 technologies, at most 200 / 600 per technology so that JavaScript doesn't swamp Rust.

The **Primary Keyword** is the candidate's own pick from a fixed list on the platform, separate from the text they wrote. It's the ground truth below and nothing upstream of the evaluation reads it. It's noisy (one label per person, self-chosen), and it says "this is their main stack", which is the right truth for a request like "C++ experts".
''')
code(r'''
graph_rows = graph_cvs()                       # [(body, text)]: body = Moreinfo + Highlights, text = Position + body
val_cvs, test_cvs = eval_cvs("val"), eval_cvs("test")
print(f"{len(graph_rows):,} CVs build the graph; {len(val_cvs):,} val and {len(test_cvs):,} test CVs carry the truth")
example = next((c for c in test_cvs if c["pk"] == "C++" and re.search("llvm", c["text"], re.I)), test_cvs[0])
print(f"\nA test CV, Primary Keyword {example['pk']}:\n")
print(scrub(example["text"][:900]))
''')

md('''
## 2. Keywords from the CVs

CVs list their skills: "Strong C, C++, bash, ARM TrustZone, gcc, llvm, clang". The vocabulary is what those lists repeat (`Vocabulary.mine`):

- **A list item** is 1–3 tokens and at most 30 characters, inside a run of at least three such items. Prose split by commas rarely produces three short pieces in a row, so "Currently, I work at …" doesn't count.
- **A keyword** is listed by at least 5 CVs. The floor is that low on purpose: LLVM is listed by exactly 5.
- **Lower-case words** ("using", "currently") must sit in lists in at least 20% of the CVs that use them; those are mostly prose.
- **Spellings merge**: react.js = reactjs = React JS, and a version folds into its keyword (C++17 → C++, Python3 → Python, .NET 6 → .NET).
- **Extraction** (`Vocabulary.extract`) finds every keyword span, so "Java Developer" names both Java Developer and Java. A one-word keyword written with spaces names only itself: "Java Script" is JavaScript, not also Java. Keywords of one or two letters (C, R, Go) count only when not written in lower case, so "ready to go" isn't Go.
''')
code(r'''
vocab = Vocabulary.mine(body for body, _ in graph_rows)
print(f"{len(vocab):,} keywords, {len(vocab.variants)} version spellings folded into them")
by_df = sorted(range(len(vocab)), key=lambda i: -vocab.df[i])
print("most listed:", ", ".join(vocab[i] for i in by_df[:40]))
print("around rank 3000:", ", ".join(vocab[i] for i in by_df[3000:3025]))
print("the long tail:", ", ".join(vocab[i] for i in by_df[-25:]))

def highlight(text, names, width=None):
    """The CV text with the named keywords marked."""
    out = html.escape(text if width is None else text[:width])
    for n in sorted(set(names), key=len, reverse=True):
        out = re.sub(r"(?<![\w+#])(" + re.escape(html.escape(n)) + r")(?![\w+#])", r"<mark>\1</mark>", out, flags=re.I)
    return out
found = [vocab[i] for i in vocab.extract(example["text"])]
print(f"\nThe example CV names {len(found)} keywords:")
display(HTML(f"<div style='font-size:13px;line-height:1.5'>{highlight(scrub(example['text']), found, 900)}</div>"))
''')

md('''
## 3. The implication graph

The API: `Cooccurrence.count` over the keyword sets of the graph-split CVs, then `subsumption_edges` and `KeywordGraph`. `build_graph(cvs)` does sections 2 and 3 in one call.

**The rule** (subsumption, Sanderson & Croft 1999, adapted): A → B, "whoever has A has B", when
- **P(B | A) ≥ 0.5**: most CVs naming A also name B;
- **n(B) ≥ 1.5 · n(A)**: B is clearly more common. Near-equivalents (HTML and CSS) get no edge, and the graph can't have cycles;
- **lift ≥ 3**: P(B | A) is at least three times B's base rate. So anything named in a third of all CVs or more (Developer, Git) can't be a target, and hubs don't absorb everything;
- **n(A, B) ≥ 5**;
- **B isn't part of A's own name**: "Unreal Engine" and "Engine" always co-occur, by construction.

No labels, no taxonomy, no LLM. `graph.verify` with `nli_judge()` (a local NLI model) can filter the edges by direction, but it isn't run here. That's about 100k NLI calls, and whether it helps is untested.
''')
code(r'''
t = time.time()
keyword_sets = [vocab.extract(text) for _, text in graph_rows]
co = Cooccurrence.count(keyword_sets, len(vocab))
contained = vocab.contained()
X = embed_keywords(vocab)                      # bge-base on the names: the trained heads and cosine read these
graph = KeywordGraph(vocab, subsumption_edges(co, contained), X)
print(f"built in {time.time() - t:.0f}s; mean {np.mean([len(s) for s in keyword_sets]):.1f} keywords per CV")
print(graph.stats())
print("\nHow the threshold on P(B | A) trades edges for confidence:")
display(Markdown(table(["P(B | A) ≥", "edges", "implied pairs (training positives)"],
    [[tau, len(e := subsumption_edges(co, contained, tau=tau)), KeywordGraph(vocab, e, X[:, :1]).stats()["closure pairs"]]
     for tau in (0.4, 0.5, 0.6, 0.7)])))
''')
code(r'''
rows = []
for n in ["LLVM", "Clang", "Qt", "STL", "Spring Boot", "Hibernate", "Redux", "Django", "Laravel", "Kubernetes", "SwiftUI", "Jetpack Compose"]:
    rows.append([n, ", ".join(f"{b} ({p:.2f})" for b, p in graph.implies(n)[:5]) or "–"])
display(Markdown("**What having a keyword implies** (direct edges, P(B | A)):\n\n" + table(["keyword", "implies"], rows)))
inc = graph.implied_by("C++")
display(Markdown(f"**{len(inc)} keywords imply C++.** The most certain 30: " + ", ".join(f"{a} ({p:.2f})" for a, p in inc[:30])))
rng = np.random.default_rng(0)
sample = [list(graph.G.edges)[i] for i in rng.choice(graph.G.number_of_edges(), 20, replace=False)]
display(Markdown("**20 edges at random**, to eyeball the noise: " + "; ".join(f"{vocab[a]} → {vocab[b]}" for a, b in sample)))
''')

md('''
## 4. The scorers

Every method scores s(A → B), "having keyword A implies having keyword B", from keyword names alone:

| method | what it is | trained |
|---|---|---|
| cosine | cos of the bge-base vectors of the two names. Symmetric. | no |
| graph lookup | the graph: 2 if A = B, 1 + P along the best path if the graph implies it, else 0; 0.01·cosine orders the rest | no |
| cos + gen | cosine + λ·generality(B), generality = log(1 + #keywords that imply B) | no |
| order + gen + cos | order embedding (Vendrov+ 2016) + λ·gen + μ·cos | on the graph |
| box + gen + cos | Gumbel box embedding (Dasgupta+ 2020) + λ·gen + μ·cos | on the graph |
| hyperbolic + gen + cos + depth | Poincaré embedding (Nickel & Kiela 2017) + λ·gen + μ·cos + μ_d·d(0, A), the depth term for the down direction | on the graph |

The heads train on the graph's implied pairs with negatives (reversed pairs, random keywords, cousins). Their configs are the methods' defaults. λ, μ and μ_d are picked on **val**, for the mean of down-unseen MAP and up-unseen MRR (defined in section 5).
''')
code(r'''
heads = {}
for h in HEADS:
    t = time.time()
    heads[h] = train(graph, h, epochs=100)
    print(f"{NAMES[h]:32} trained in {time.time() - t:.0f}s on {len(graph.positives):,} implied pairs")
val = CVIndex(val_cvs, vocab)
LM = [(l, m) for l in (0, 0.1, 0.3, 1) for m in (0, 1, 3, 10)]
w = {"cosgen": tune(val, lambda o: cosine_generality(graph.X, graph.generality, *o), [(l,) for l in (0, 0.01, 0.03, 0.1, 0.3, 1)])[0]}
for h in ("order", "box"):
    w[h] = tune(val, lambda o, h=h: hybrid(heads[h].scorer(graph.X), graph, *o), LM)[0]
def with_depth(o):
    heads["hyp"].mu = o[2]
    return hybrid(heads["hyp"].scorer(graph.X), graph, *o[:2])
w["hyp"] = tune(val, with_depth, [(l, m, d) for l, m in LM for d in (0, 0.5, 1, 1.5)])[0]
S = scorers(graph, heads, w)
print("\nweights picked on val:", {NAMES[k]: v for k, v in w.items()})
''')

md('''
## 5. The evaluation grid (test)

The truth is the Primary Keyword p, for the 18 technologies. "Names p" means the CV text names p's keyword. Aliases count: JS for JavaScript, Go for Golang, node for Node.js, versions like C++17.

| query type | the question | candidates | relevant | metric |
|---|---|---|---|---|
| **down seen** | "who are the C++ people?", among CVs that say C++ | test CVs naming p | Primary Keyword p | MAP, averaged over p |
| **down unseen** | the same, among CVs that never say C++ | test CVs not naming p | Primary Keyword p | MAP |
| **up seen** | "what is this person's main stack?", for a CV that names it | the 18 keywords | its own p | MRR, accuracy@1 |
| **up unseen** | the same, for a CV that never names its own stack | the 18 keywords | its own p | MRR, accuracy@1 |
| **identity** | does a keyword come first for itself? | all keywords | the keyword itself | share at rank 1 (keyword level); AUC of CVs naming p over the rest (CV level) |

A CV's score toward p is the best score of its keywords toward p's keyword. **Keyword search** is the baseline: score 1 if the CV names p, else 0, with ties broken at random. It is perfect at finding who names p, and blind to everyone else. Random ranking scores the share of relevant CVs.

How often the text doesn't name the candidate's own Primary Keyword, the people "down unseen" is about:
''')
code(r'''
test = CVIndex(test_cvs, vocab)
T = targets(vocab)
pk = np.array([c["pk"] for c in test_cvs])
display(Markdown(table(["Primary Keyword", "test CVs", "text doesn't name it"],
    [[p, int((pk == p).sum()), f"{(~test.names(T[p][1]).numpy()[pk == p]).mean():.0%}"] for p in TECH])))
''')
code(r'''
t = time.time()
results = {"keyword search": grid(test, None)}
for k, M in S.items(): results[NAMES[k]] = grid(test, M)
cols = ["down seen MAP", "down unseen MAP", "up seen MRR", "up seen acc@1", "up unseen MRR", "up unseen acc@1", "identity: keyword top 1", "identity: CV AUC"]
display(Markdown(table(["method"] + cols, [[k] + [r.get(c, "–") for c in cols] for k, r in results.items()])))
chance = np.mean([np.mean(pk[~test.names(T[p][1]).numpy()] == p) for p in TECH])
print(f"scored in {time.time() - t:.0f}s; random ranking on down unseen ≈ {chance:.3f}")
''')
code(r'''
R = results
scored = [k for k in R if k != "keyword search"]
quality = cols[:6]
best = lambda c: max(scored, key=lambda k: R[k][c])
g, kw, cos = R["graph lookup"], R["keyword search"], R["cosine"]
wins = [c for c in quality if best(c) == "graph lookup"]
trained = [NAMES[k] for k in ("order", "box", "hyp")]
beat = [(c, best(c)) for c in quality if best(c) in trained]
zero = [NAMES[k] for k, v in w.items() if v[0] == 0]
weak = [k for k in scored if R[k]["identity: keyword top 1"] < 0.9]
display(Markdown(f"""
**What the grid says**

- **Keyword search is blind to the unseen half.** Down unseen {kw['down unseen MAP']:.3f} is chance level ({chance:.3f}). The best scorer there is {best('down unseen MAP')} with {R[best('down unseen MAP')]['down unseen MAP']:.3f}, about {R[best('down unseen MAP')]['down unseen MAP'] / chance:.0f}× chance. Finding a C++ person who never says C++ is hard, and the numbers are low in absolute terms.
- **The graph itself, with no training, is the best scorer on {len(wins)} of {len(quality)} columns**: {g['down seen MAP']:.3f} / {g['down unseen MAP']:.3f} down MAP and {g['up seen acc@1']:.3f} / {g['up unseen acc@1']:.3f} up accuracy@1, against cosine's {cos['down seen MAP']:.3f} / {cos['down unseen MAP']:.3f} and {cos['up seen acc@1']:.3f} / {cos['up unseen acc@1']:.3f}. {"A trained model is best on " + "; ".join(f"{c} ({k}, {R[k][c]:.3f} vs graph {g[c]:.3f})" for c, k in beat) + "." if beat else "No trained model beats it on any column."} The trained embeddings learn from this graph, and every test keyword is in the same closed vocabulary, so they have little to generalize to. Where they could matter, keywords the graph has no edge for, is untested.
- **Generality:** val picked λ = 0 for {", ".join(zero) if zero else "no method"}{" (so cos + gen ranks exactly like cosine)" if "cos + gen" in zero else ""}.
- **Identity:** {(", ".join(f"{k} ({R[k]['identity: keyword top 1']:.2f})" for k in weak) + " doesn't rank every keyword first for itself. That doesn't hurt search, because search lists the CVs that name the requirement first anyway (section 6).") if weak else "every scorer ranks each keyword first for itself, and search lists the CVs that name the requirement first anyway (section 6)."}
"""))
''')

md('''
## 6. Search, and the HR demo

`Search(index, scorers).query(text, mode, method)` reads the keywords a request names (no LLM), **longest match first**: "Need a Spring Boot developer" is Spring Boot, not also Spring and Boot. Keywords that a third or more of the CVs name, like "Developer", match too many CVs to be a requirement and are dropped. That's the same bound that keeps them from being graph targets. Then it ranks the CVs:

1. **CVs that name the requirement come first**, whatever the mode. This is the identity correspondence: a CV that says LLVM answers "LLVM" before any CV that only implies it.
2. **The rest are a fallback**, scored by their best keyword plus 0.1 × the mean of their three best, so more support counts:
   - **down**, "who has it": s(CV keyword → requirement). Qt, STL and LLVM people for "C++".
   - **up**, "what it implies": s(requirement → CV keyword), plus the down score. A C++ CV is the closest thing to an LLVM CV, and among C++ CVs the ones that also look like LLVM people (Clang, GCC) come first, not every CV that mentions C++ once.

The search runs over the 6.7k test CVs. The Primary Keyword column is the truth, shown only to check the results.
''')
code(r'''
search = Search(test, S)

def show(text, mode, method, named=3, implied=8, expect=None):
    req, top = search.query(text, mode, method, k=named, names=True)
    _, rest = search.query(text, mode, method, k=implied, names=False)
    rows = []
    for i, r in enumerate(top + rest, 1):
        c, why = {**r["cv"], "text": scrub(r["cv"]["text"])}, r["because"][0]
        how = f"names {', '.join(req)}" if r["names"] else f"via <b>{html.escape(why or '–')}</b>"
        hit = "✓" if expect and c["pk"] == expect else ""
        m = re.search(re.escape(why or ""), c["text"], re.I) if why else None
        lo = max(0, m.start() - 70) if m else 0
        rows.append(f"<tr><td>{i}</td><td>{how}</td><td>{html.escape(c['position'][:45])}</td><td>{c['pk']} {hit}</td>"
                    f"<td style='font-size:12px'>…{highlight(c['text'][lo:lo + 180], [why] if why else [])}…</td></tr>")
    hits = sum(r["cv"]["pk"] == expect for r in rest) if expect else None
    head = f"<b>{NAMES[method]}</b>, {mode}, requirement: {', '.join(req) or 'none found'}" + \
           (f" — {hits}/{len(rest)} of the fallback have Primary Keyword {expect}" if expect else "")
    display(HTML(f"<p>{head}</p><table><tr><th>#</th><th>why</th><th>title</th><th>Primary Keyword</th><th>where</th></tr>{''.join(rows)}</table>"))
''')

md('''
### "Would be nice if has LLVM knowledge" (up)

LLVM is rare, so only a few test CVs name it. They come first; after them comes the fallback, CVs with what LLVM implies.
''')
code(r'''
for m in ("graph", "box", "hyp", "cos"):
    show("Would be nice if has LLVM knowledge", "up", m, named=4, implied=6, expect="C++")
''')
md('''
**Checking the up-mode fallback.** Up mode has no ground truth, so here is a check on 22 hand-picked requests whose fallback people are obvious: Spring → Java people, Django → Python people, SwiftUI → iOS people, and so on (`search.UP_CHECK`). For each request, the share of the first 20 fallback CVs (those that don't name it) whose Primary Keyword is the obvious one. The pairs are my picks, not data.
''')
code(r'''
from cvsearch import UP_CHECK, up_check
checks = {m: up_check(search, m) for m in ("graph", "hyp", "box", "order", "cos")}
asked = list(checks["graph"])
display(Markdown(table(["request → obvious people"] + [NAMES[m] for m in checks],
    [[f"{r} → {dict(UP_CHECK)[r]}"] + [f"{checks[m].get(r, float('nan')):.0%}" for m in checks] for r in asked] +
    [["**mean**"] + [f"**{np.mean(list(checks[m].values())):.0%}**" for m in checks]])))
''')
code(r'''
lines = []
for m in ("graph", "box", "hyp", "cos"):
    _, rest = search.query("Would be nice if has LLVM knowledge", "up", m, k=20, names=False)
    via = [r["because"][0] for r in rest]
    lines.append(f"- **{NAMES[m]}**: the first 20 fallback CVs come via {', '.join(dict.fromkeys(via))}; "
                 f"{sum(r['cv']['pk'] == 'C++' for r in rest)}/20 have Primary Keyword C++.")
display(Markdown("\n".join(lines) + f"""

The graph's edges out of LLVM: {", ".join(f"{b} ({p:.2f})" for b, p in graph.implies("LLVM")) or "none"}. Cosine reads names only, so its fallback follows spelling. A hyperbolic distance is symmetric, so central hubs are close to everything, and its depth term only acts in down mode. There is no ground truth for up-mode fallbacks, so this part is an eyeball check, not a measurement.
"""))
''')

md('''
### "Looking for the C++ experts" (down)

Hundreds of test CVs say C++. Search lists them first; the interesting part is the fallback, the CVs that never say C++.
''')
code(r'''
for m in ("graph", "hyp", "cos"):
    show("Looking for the C++ experts", "down", m, named=3, implied=10, expect="C++")
''')
code(r'''
prec = {}
for m in S:
    _, rest = search.query("Looking for the C++ experts", "down", m, k=20, names=False)
    prec[NAMES[m]] = np.mean([r["cv"]["pk"] == "C++" for r in rest])
base = np.mean(pk[~test.names(T["C++"][1]).numpy()] == "C++")
display(Markdown(f"**Share of C++ people in the first 20 fallback CVs**, against {base:.1%} among all CVs that don't say C++: " +
                 ", ".join(f"{k} {v:.0%}" for k, v in prec.items())))
''')

md('''
## 7. What this shows, and what it doesn't

- **The whole pipeline needs only CVs.** Mined keywords, a co-occurrence graph and graph lookup take keyword search from chance level to a usable fallback on people who never wrote the keyword. Nothing in it is trained or labelled.
- **On this collection the graph is the method.** The embedding heads learn from the graph and don't beat it, because every test keyword is in the same closed vocabulary. Their potential case, keywords the graph has no edge for, is untested.
- **"Experts" and "would be nice" aren't modelled.** Search finds who has a skill, not how well or how much it's wanted. Within the CVs that name C++, the order comes only from how many of their other keywords support it.
- **The truth is coarse.** One self-picked Primary Keyword per CV, for 18 technologies. It says nothing about LLVM → C++ as such. Up-mode fallbacks are checked only on 22 hand-picked requests.
- **The vocabulary is noisy** (see the long tail and the random edges in sections 2 and 3). With a lift of 3 and 5 co-occurrences required, most of that noise never forms an edge.
- **Not run:** the NLI edge filter (`verify` with `nli_judge`), seeds beyond 0, and serving from Qdrant.
''')
md('''
## 8. The demo page

`cvsearch.demo.export` writes the search as one self-contained HTML page (`data/djinni/cvsearch_demo.html`, from the template `tools/cvsearch_demo.html`). A static page can't run the models, so it stores, for the 1,000 keywords most named in the test CVs plus the demo ones, the top CVs per mode and method from the scorers above. It parses free text with the same keyword rules, in JavaScript.
''')
code(r'''
from pathlib import Path
from cvsearch.demo import export
page = Path("data/djinni/cvsearch_demo.html")
print(export(page, test, graph, {k: S[k] for k in ("graph", "hyp", "box", "order", "cos")}, NAMES, results), "→", page)
print(f"notebook ran in {(time.time() - t0) / 60:.0f} minutes")
''')

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
json.dump(nb, open(out, "w"), indent=1)
print(f"wrote {out}: {len(cells)} cells")
