"""Builds methods.ipynb: every method trained and served from Qdrant, one section each.
Usage: python3 tools/build_methods.py methods.ipynb, then execute it with nbconvert (Qdrant on :6333)."""
import json
import sys

out = sys.argv[1]
cells = []
def md(s): cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip()})
def code(s): cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip()})

md('''
# Every method, trained and served from Qdrant

s(A→B) is how strongly having skill A implies having skill B. Someone with Spring Boot has Java, and someone with Java may never have used Spring Boot, so a good s is asymmetric. This notebook trains each method from the spike and serves it from Qdrant. Each section shows the method's code as it is in `skillmatch/`, trains it with the config picked on validation, loads it into a Qdrant collection, queries it, and checks that the Qdrant ranking gives the same test MAP as the torch ranking.

The comparison between methods, and what the numbers mean, is in `showcase.ipynb` and `docs/results.md`. This notebook covers the mechanics.

Two directions:

- Up: given a skill, which skills does it imply? Task A asks it for skills never seen in training, Task B for implications hidden from the training graph. Sections 1 to 11.
- Down: given a requirement, who has it? Task D. Section 12.

Qdrant serves a score in one of three ways, and the method decides which:

| Route | When | Methods |
|---|---|---|
| Native distance with HNSW | the score is a dot product or an L1 distance between two stored vectors | cosine, dual encoder, TransE, distilled dual |
| Formula Query over a cosine prefetch | Qdrant has no distance for the geometry, and the score is a short closed form of numbers stored in the payload | cos + gen, order, hyperbolic, any head + gen + cos |
| Shortlist from Qdrant, scored in the client | the score is a neural network, or too long for a formula | box, pair MLP, cross-encoder, LLM judge |

Before running: start Qdrant (`podman run -d -p 6333:6333 docker.io/qdrant/qdrant`) and run from the repo root. The whole notebook takes about 20 minutes on a laptop, mostly for the cross-encoder training and for the Formula Query checks, which rescore the full catalogue on every query.

All numbers here come from one seed (0). The "stored" columns come from `results/` for reference: three-seed means for the trained heads, so they differ by seed noise, one earlier run for the cross-encoder, and the cached answers for the LLM judges, the same answers this notebook reads.
''')
code(r'''
import inspect
import time

import numpy as np
import torch
from IPython.display import Markdown, display
from qdrant_client import QdrantClient, models

from experiments.zoo import FOLDED, config, load, seed_all
from skillmatch import formulas, qdrant
from skillmatch.data import DEVICE, SkillGraph
from skillmatch.methods import (BoxEmbedding, CrossEncoder, DualEncoder, OrderEmbedding, PairMLP, PoincareEmbedding,
                                TransE, cosine, cosine_generality, distill, folded_vectors, hybrid)
from skillmatch.methods.cosine import LAMBDA
from skillmatch.methods.hybrid import tune
from skillmatch.methods.llm_judge import Judged, judge_all, test_tasks
from skillmatch.tasks import down, evaluate, expanded
from skillmatch.training import labelled_pairs, negatives, train_pairwise, train_softmax

graph = SkillGraph()
client = QdrantClient("localhost", port=6333, timeout=300)
ALL = len(graph.catalogue)  # prefetch the whole catalogue, so Qdrant and torch rank the same candidates
COSINE, DOT = models.Distance.COSINE, models.Distance.DOT
DEMO = graph.by_label("Express.js")  # a held-out test skill: it is in no training edge
print(f"{len(graph.ids)} skills, catalogue {ALL}, {len(graph.positives)} training pairs, demo query: {graph.label(DEMO)}")
''')
code(r'''
STORED = load("recall.json")  # three-seed means of the trained heads
PARITY = {}


def source(*objects):
    """The code of functions or classes, as it is in the repo."""
    display(Markdown("\n".join(f"```python\n{inspect.getsource(o)}```" for o in objects)))


def table(rows):
    """{row: {column: number}} → Markdown table."""
    cols = list(next(iter(rows.values())))
    cell = lambda x: "" if x is None else f"{float(x):.3f}"
    lines = ["| | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
    display(Markdown("\n".join(lines + [f"| {n} | " + " | ".join(cell(r.get(c)) for c in cols) + " |" for n, r in rows.items()])))


def top(search, a=DEMO, k=5):
    """The k best hits of a Qdrant search for skill a."""
    return ", ".join(f"{graph.label(h.id)} ({h.score:.2f})" for h in search(a)[:k])


def rescored(hits, scores):
    """Qdrant hits with scores computed in the client, best first."""
    return sorted((h.model_copy(update={"score": s}) for h, s in zip(hits, scores)), key=lambda h: -h.score)


def check(name, M, search, stored=None):
    """Test MAP of Tasks A and B: the torch scorer M, and the same method served by Qdrant (search)."""
    with torch.no_grad():
        t, q = (evaluate(graph, S, "test", directions=False) for S in (M, qdrant.served_up(search)))
    stored = stored or (STORED[name]["A MAP"], STORED[name]["B MAP"])
    PARITY[name] = {"A torch": t["A"]["MAP"], "A Qdrant": q["A"]["MAP"], "B torch": t["B"]["MAP"], "B Qdrant": q["B"]["MAP"],
                    "A stored": stored[0], "B stored": stored[1]}
    table({name: PARITY[name]})
''')

md('''
## 1. Cosine

s(A→B) = cos(x_A, x_B), where x is the frozen bge-base vector of the skill's text. It gives the same number in both directions, so it is the floor every other method has to beat. Nothing to train.

Qdrant: one dense vector per skill with Cosine distance. The queries here use the encoder rows cached in `data/`, the same vectors the heads read. In production the query text would go through Qdrant inference (`qdrant.document(text)`, section 12).
''')
code(r'''
source(cosine)
''')
code(r'''
qdrant.catalogue_collection(client, "skills_text", graph, {"text": (graph.X, COSINE)})


def cosine_search(a, limit=ALL):
    return client.query_points("skills_text", query=graph.X[a].tolist(), using="text", limit=limit).points


print("Express.js →", top(cosine_search))
check("cosine", cosine(graph.X), cosine_search)
''')

md('''
## 2. Cosine + generality

s(A→B) = cos(A, B) + λ·gen(B), with gen(B) = log(1 + number of training skills that imply B). Requirements tend to be general skills, and a count of known descendants is a cheap signal of that. λ = 0.05 was picked on validation. Still nothing to train.

Qdrant: `catalogue_collection` stores gen in each point's payload. A Formula Query adds it to the cosine score of a prefetch (`$score`). The optional filter drops skills by id, which the rerankers in sections 10 and 11 need.
''')
code(r'''
source(cosine_generality, formulas.cos_gen)
''')
code(r'''
def cos_gen_search(a, limit=ALL, exclude=()):
    return client.query_points(
        "skills_text",
        prefetch=models.Prefetch(query=graph.X[a].tolist(), using="text", limit=ALL),
        query=models.FormulaQuery(formula=formulas.cos_gen(LAMBDA)),
        query_filter=models.Filter(must_not=[models.HasIdCondition(has_id=list(exclude))]) if exclude else None,
        limit=limit).points


print("Express.js →", top(cos_gen_search))
check("cos+gen", cosine_generality(graph.X, graph.generality), cos_gen_search)
''')

md('''
## 3. Dual encoder

Two small networks on the same text vector, one per role: s(A→B) = src(A)·tgt(B). It is asymmetric because src and tgt differ.

Training (`train_pairwise`, shared by the dual, box, order, TransE and pair MLP heads): each epoch takes every pair (A, B) of the training closure as a positive, plus up to three negatives: the reversed pair (B, A), a random catalogue skill A doesn't imply, and, when there is one, a cousin (another skill below B that A doesn't imply). The head's own loss scores them; for the dual it is binary cross-entropy on the dot product.
''')
code(r'''
source(DualEncoder, negatives, train_pairwise)
''')
code(r'''
kw, epochs, lr = config("dual")  # picked on val: results/sweep_best.json
print(kw, epochs, lr)
seed_all(0)
dual = train_pairwise(DualEncoder(**kw), graph, epochs, lr)
''')
md('''
Qdrant: two named vectors per skill, both with Dot distance. The up search sends src(A) against the stored tgt vectors. The same collection answers the reverse question (which skills imply B?) by sending tgt(B) against the stored src vectors.
''')
code(r'''
with torch.no_grad(): src, tgt = dual.src(graph.X), dual.tgt(graph.X)
qdrant.catalogue_collection(client, "skills_dual", graph, {"src": (src, DOT), "tgt": (tgt, DOT)})


def dual_search(a, limit=ALL):  # what does a imply?
    return client.query_points("skills_dual", query=src[a].tolist(), using="tgt", limit=limit).points


def dual_implied_by(b, limit=5):  # which catalogue skills imply b?
    return client.query_points("skills_dual", query=tgt[b].tolist(), using="src", limit=limit).points


print("Express.js →", top(dual_search))
print("→ JavaScript:", ", ".join(graph.label(h.id) for h in dual_implied_by(graph.by_label("JavaScript"))))
check("dual", dual.scorer(graph.X), dual_search)
''')

md('''
## 4. Box embeddings, + gen + cos

Each skill is an axis-aligned box with soft (Gumbel) edges, and A implies B when A's box sits inside B's: s(A→B) = log P(B | A) = log vol(A ∩ B) − log vol(A), at most 0. Trained with `train_pairwise`; the loss is −log P on positives and −log(1 − P) on negatives.

The heads learn from under a thousand edges and lose some of what the text encoder knows, so every head also comes in a hybrid, head + λ·gen + μ·cos, with (λ, μ) picked on validation (`tune`).
''')
code(r'''
source(BoxEmbedding, hybrid, tune)
''')
code(r'''
kw, epochs, lr = config("box")  # results/sweep_best.json
seed_all(0)
box = train_pairwise(BoxEmbedding(**kw), graph, epochs, lr)
with torch.no_grad(): box_w = tune(graph, box.scorer(graph.X))
print(kw, epochs, lr, "(λ, μ) =", box_w)
''')
md('''
Qdrant has no box distance. The box score can be written as a Formula Query (`formulas.box`, timed below), but at 128 dimensions the formula is about 550 KB of JSON and took about 14 s per query over the full catalogue. So Qdrant returns a cosine shortlist with each skill's box corners in the payload, and the client scores it with the model's own `score`.
''')
code(r'''
with torch.no_grad(): corners = box.embed(graph.X)  # [lo | hi] per skill
qdrant.catalogue_collection(client, "skills_box", graph, {"text": (graph.X, COSINE)}, payload={"box": corners})


def box_search(a, lam=0.0, mu=0.0, limit=ALL):
    """Cosine shortlist from Qdrant, scored here: box score + λ·gen + μ·cos."""
    hits = client.query_points("skills_box", query=graph.X[a].tolist(), using="text", limit=limit,
                               with_payload=["box", "gen"]).points
    stored = torch.tensor([h.payload["box"] for h in hits], device=DEVICE)
    gen, cos = (torch.tensor(v, device=DEVICE) for v in zip(*[(h.payload["gen"], h.score) for h in hits]))
    with torch.no_grad(): s = box.score(corners[a], stored) + lam * gen + mu * cos
    return rescored(hits, s.tolist())


print("Express.js →", top(lambda a: box_search(a, *box_w)))
check("box+gen+cos", hybrid(box.scorer(graph.X), graph, *box_w), lambda a: box_search(a, *box_w))
''')
code(r'''
source(formulas.box, formulas.logaddexp, formulas.softplus)
''')
code(r'''
# The same box + gen + cos as a Formula Query on a 100-skill prefetch, against the client-side route on the same 100
formula = formulas.hybrid(formulas.box(box, corners[DEMO]), *box_w)
t = time.perf_counter()
server = client.query_points("skills_box", prefetch=models.Prefetch(query=graph.X[DEMO].tolist(), using="text", limit=100),
                             query=models.FormulaQuery(formula=formula), limit=10).points
t_server = time.perf_counter() - t
t = time.perf_counter()
local = box_search(DEMO, *box_w, limit=100)[:10]
t_local = time.perf_counter() - t
print(f"Formula Query: {t_server:.2f} s, client-side: {t_local:.3f} s, same top 10: {[h.id for h in server] == [h.id for h in local]}, "
      f"max score gap {max(abs(s.score - l.score) for s, l in zip(server, local)):.1e}")
''')

md('''
## 5. Order embeddings, + gen + cos

Each skill is a point with non-negative coordinates, specific skills further from the origin, and A implies B when A ≥ B on every coordinate. s(A→B) = −Σᵢ max(0, f(B)ᵢ − f(A)ᵢ)², the order violation, which is 0 when A sits above B everywhere. Trained with `train_pairwise` and a max-margin loss. Order + gen + cos has the best Task A MAP of the trained heads, 0.618 over three seeds.

Qdrant: f(B) goes in the payload as an array, and a Formula Query computes the violation over a cosine prefetch. The max becomes (x + |x|) / 2, since formulas have abs but no max. The head, gen and cos are then one server-side query.
''')
code(r'''
source(OrderEmbedding, formulas.order, formulas.relu, formulas.hybrid)
''')
code(r'''
kw, epochs, lr = config("order")  # results/compare_sweep.json
seed_all(0)
order = train_pairwise(OrderEmbedding(**kw), graph, epochs, lr)
with torch.no_grad(): order_w = tune(graph, order.scorer(graph.X))
print(kw, epochs, lr, "(λ, μ) =", order_w)
''')
code(r'''
with torch.no_grad(): f_order = order.embed(graph.X)
qdrant.catalogue_collection(client, "skills_order", graph, {"text": (graph.X, COSINE)}, payload={"order": f_order})


def order_search(a, lam=0.0, mu=0.0, limit=ALL, prefetch=ALL):
    return client.query_points(
        "skills_order",
        prefetch=models.Prefetch(query=graph.X[a].tolist(), using="text", limit=prefetch),
        query=models.FormulaQuery(formula=formulas.hybrid(formulas.order(f_order[a]), lam, mu)),
        limit=limit).points


print("Express.js →", top(lambda a: order_search(a, *order_w)))
check("order+gen+cos", hybrid(order.scorer(graph.X), graph, *order_w), lambda a: order_search(a, *order_w))
''')

md('''
## 6. Hyperbolic embeddings, with the is-a score

Poincaré embeddings (Nickel and Kiela 2017), the geometry of the Qdrant hyperbolic article. Skills are points in the unit ball and general skills settle near the origin. The distance d is symmetric, so the article's score −d has no direction. The is-a score adds one: s(A→B) = −(1 + α(‖B‖ − ‖A‖))·d(A, B) also asks B to sit closer to the origin than A. α = 0.1 was picked on validation.

Training (`train_softmax`, the Nickel and Kiela loss): a softmax over −d that has to pick B out of B plus ten skills unrelated to A.

Qdrant: the coordinates go in the payload with ‖B‖² and ‖B‖, and the formula writes acosh as ln(x + √(x² − 1)), as in the article.
''')
code(r'''
source(PoincareEmbedding, train_softmax, formulas.poincare)
''')
code(r'''
kw, epochs, lr = config("hyp+isa")  # results/compare_sweep.json; alpha only changes the score, not the training
seed_all(0)
hyp = train_softmax(PoincareEmbedding(**kw), graph, epochs, lr)
print(kw, epochs, lr)
''')
code(r'''
with torch.no_grad(): p_hyp = hyp.embed(graph.X)
qdrant.catalogue_collection(client, "skills_hyp", graph, {"text": (graph.X, COSINE)},
                            payload={"hyp": p_hyp, "hyp_sq": (p_hyp ** 2).sum(-1).clamp(max=1 - 1e-5), "hyp_norm": p_hyp.norm(dim=-1)})


def hyp_search(a, limit=ALL):
    return client.query_points(
        "skills_hyp",
        prefetch=models.Prefetch(query=graph.X[a].tolist(), using="text", limit=ALL),
        query=models.FormulaQuery(formula=formulas.poincare(hyp, p_hyp[a])),
        limit=limit).points


print("Express.js →", top(hyp_search))
check("hyp+isa", hyp.scorer(graph.X), hyp_search)
''')

md('''
## 7. TransE, + gen + cos

One relation r, "implies": s(A→B) = −‖f(A) + r − f(B)‖₁ (the L1 norm was picked on validation). Trained with `train_pairwise` and a max-margin loss.

Qdrant: TransE alone is a native search. Store f(B) with Manhattan distance and send f(A) + r. Qdrant returns the distance, smaller first, so the search negates it to get s. The hybrid also needs gen and cos, so it keeps f(B) in the payload as well and computes the distance in a formula over a cosine prefetch.
''')
code(r'''
source(TransE, formulas.transe)
''')
code(r'''
kw, epochs, lr = config("transe")  # results/compare_sweep.json
seed_all(0)
transe = train_pairwise(TransE(**kw), graph, epochs, lr)
with torch.no_grad(): transe_w = tune(graph, transe.scorer(graph.X))
print(kw, epochs, lr, "(λ, μ) =", transe_w)
''')
code(r'''
with torch.no_grad(): f_transe = transe.embed(graph.X)
distance = models.Distance.MANHATTAN if transe.p == 1 else models.Distance.EUCLID
qdrant.catalogue_collection(client, "skills_transe", graph, {"text": (graph.X, COSINE), "transe": (f_transe, distance)},
                            payload={"transe": f_transe})


def transe_search(a, limit=ALL):
    with torch.no_grad(): query = f_transe[a] + transe.r
    hits = client.query_points("skills_transe", query=query.tolist(), using="transe", limit=limit).points
    return [h.model_copy(update={"score": -h.score}) for h in hits]  # Qdrant returns the distance


def transe_gen_cos_search(a, lam, mu, limit=ALL):
    return client.query_points(
        "skills_transe",
        prefetch=models.Prefetch(query=graph.X[a].tolist(), using="text", limit=ALL),
        query=models.FormulaQuery(formula=formulas.hybrid(formulas.transe(f_transe[a], transe), lam, mu)),
        limit=limit).points


print("Express.js →", top(transe_search))
check("transe", transe.scorer(graph.X), transe_search)
check("transe+gen+cos", hybrid(transe.scorer(graph.X), graph, *transe_w), lambda a: transe_gen_cos_search(a, *transe_w))
''')

md('''
## 8. Pair MLP

A classifier on the ordered pair: MLP([x_A, x_B, x_A ⊙ x_B, x_A − x_B]) → the logit of P(B | A). It has no vector per skill, so Qdrant can only supply candidates: a cosine shortlist with the stored text vectors, scored in the client. With the whole catalogue as the shortlist, the ranking equals the torch one. A deployment would score a shorter list: the cos + gen top 50 holds 99% of the Task A answers and 93% of the Task B answers.
''')
code(r'''
source(PairMLP)
''')
code(r'''
kw, epochs, lr = config("pair_mlp")  # results/compare_sweep.json
seed_all(0)
mlp = train_pairwise(PairMLP(**kw), graph, epochs, lr)
print(kw, epochs, lr)
''')
code(r'''
def pair_mlp_search(a, limit=ALL):
    hits = client.query_points("skills_text", query=graph.X[a].tolist(), using="text", limit=limit, with_vectors=["text"]).points
    xb = torch.tensor([h.vector["text"] for h in hits], device=DEVICE)
    with torch.no_grad(): s = mlp(graph.X[a].expand(len(xb), -1), xb)
    return rescored(hits, s.tolist())


print("Express.js →", top(pair_mlp_search))
check("pair_mlp", mlp.scorer(graph.X), pair_mlp_search)
''')

md('''
## 9. Distilled dual, with gen + cos folded in

Box + gen + cos ranks well, but it needs a shortlist and client-side scoring. Distillation trains a dual encoder to reproduce its ranking: for each catalogue skill, the student's softmax over the catalogue matches the teacher's (listwise KL). Only catalogue skills are queries and candidates, so held-out skills and hidden edges never reach the student.

With the fold, the student learns only the box part and keeps gen and cos from the teacher's (λ, μ). The score stays one dot product, [src(A), √μ·x_A, 1] · [tgt(B), √μ·x_B, λ·gen(B)], so Qdrant serves it natively with Dot distance.
''')
code(r'''
source(distill, folded_vectors)
''')
code(r'''
seed_all(0)
teacher = hybrid(box.scorer(graph.X), graph, *box_w)  # box + gen + cos from section 4
student, folded = distill(graph, teacher, **FOLDED, fold=box_w)
print(FOLDED, "fold (λ, μ) =", box_w)
''')
code(r'''
with torch.no_grad(): query, document = folded_vectors(student, graph.X, graph.generality, *box_w)
qdrant.catalogue_collection(client, "skills_folded", graph, {"folded": (document, DOT)})


def folded_search(a, limit=ALL):
    return client.query_points("skills_folded", query=query[a].tolist(), using="folded", limit=limit).points


print("Express.js →", top(folded_search))
check("distilled dual, folded", folded, folded_search)
''')

md('''
## 10. Cross-encoder reranker

`bge-reranker-base` fine-tuned on the same labelled pairs as the heads. It reads the two texts together, in order, so it has direction, but it costs one transformer pass per pair and can only rerank a shortlist.

Qdrant supplies the shortlist: the cos + gen top 50, minus the skills the training graph already says A implies (a `HasIdCondition` filter). It is the same shortlist the LLM judges rerank in section 11. The 50 go first, ordered by cos + gen + p(A→B), the blend `Judged` uses; the rest keep their cos + gen order.
''')
code(r'''
def known(a):
    """a itself and the skills the training graph says a implies: Task B filters them out."""
    return graph.ancestors_train.get(a, set()) | {a}


def shortlist(a, k=50):
    return [h.id for h in cos_gen_search(a, limit=k, exclude=known(a))]


def judged_search(p):
    """p: {A: {B: p(A→B)}} for the shortlist of each A. Qdrant's cos + gen list with the judged skills on top."""
    def search(a):
        hits, pa = cos_gen_search(a, exclude=known(a)), p.get(a, {})
        return rescored(hits, [h.score + 10 + pa[h.id] if h.id in pa else h.score for h in hits])
    return search


tasks, _ = test_tasks(graph)  # the torch shortlists behind results/llm_judge.json
same = sum(set(shortlist(a)) == set(c) for a, c in tasks)
print(f"Qdrant shortlist = torch shortlist for {same} of {len(tasks)} test queries")
assert same == len(tasks)
''')
code(r'''
source(CrossEncoder)
''')
code(r'''
cfg, epochs, lr = load("compare.json")["cross_enc"]["config"]  # epochs picked on val (experiments/compare.py ce)
print(cfg, epochs, lr)
seed_all(0)
ce = CrossEncoder([graph.text(i) for i in range(len(graph.ids))], cfg["model"], cfg["max_len"])
t = time.time()
for _ in range(epochs): ce.fit_epoch(labelled_pairs(graph), lr)
print(f"trained in {time.time() - t:.0f} s")
''')
code(r'''
def p_cross_encoder(a, ids):
    return torch.sigmoid(ce.pairs(torch.tensor([a] * len(ids)), torch.tensor(ids))).tolist()


ce_p = {a: dict(zip(ids, p_cross_encoder(a, ids))) for a, _ in tasks for ids in [shortlist(a)]}
stored = load("compare.json")["cross_enc rerank50"]
check("cross-encoder, top 50", Judged(graph, ce_p, {}), judged_search(ce_p), (stored["A"]["MAP"], stored["B"]["MAP"]))
''')

md('''
## 11. LLM judges

Two LLMs from different model families score p(A→B) for the same shortlists: Claude Sonnet 5 and Codex. The prompt is below. Claude wrote the graph's labels, so a Claude judge alone would be circular. The answers are cached in `data/llm_cache/` by prompt hash, so this cell makes no LLM calls. The reranking is the same `Judged` blend as for the cross-encoder.
''')
code(r'''
from skillmatch.methods.llm_judge import rank_prompt
print(rank_prompt(graph, tasks[:1])[:1500])
''')
code(r'''
llm = load("llm_judge.json")
for judge in ("claude-sonnet-5", "codex-gpt-6-astra"):
    rank, pair, missing = judge_all(graph, judge)
    check(f"LLM {judge}, top 50", Judged(graph, rank, pair), judged_search(rank), (llm[judge]["A"]["MAP"], llm[judge]["B"]["MAP"]))
''')

md('''
## 12. Down: who has a requirement

Every section above answers the up question. A recruiter asks the down question: given requirement B, rank the profiles by s(profile→B). Scoring every profile directly by s(profile→B) gave test MAP between 0.02 and 0.37 (Task D in `docs/results.md`). Ingest-time expansion scored higher for every model: when a profile arrives, the scorer lists its top K implied catalogue skills once, and Qdrant stores the list as a sparse vector (index = skill id, value = 926 − rank). A requirement is then a sparse query {b: 1}, and several requirements add up. Ingesting one CV rescores nobody.

This route also uses Qdrant inference: `qdrant.document(text)` hands the text to Qdrant, which embeds it with FastEmbed (inside the client here, on the server in Qdrant Cloud). The profiles are the 925 catalogue skills plus the 110 held-out test skills.
''')
code(r'''
source(qdrant.expansions, qdrant.ingest_profiles, qdrant.link, qdrant.who_has)
''')
code(r'''
profiles = graph.catalogue + graph.heldout["test"]
order_gen_cos = hybrid(order.scorer(graph.X), graph, *order_w)  # section 5
t = time.time()
with torch.no_grad(): qdrant.build(client, graph, {"order+gen+cos": order_gen_cos}, [100], profiles)
print(f"ingested {len(profiles)} profiles in {time.time() - t:.0f} s")

skills = qdrant.link(client, "JavaScript")  # requirement text → catalogue skill
print("linked to:", graph.label(skills[0]))
for h in qdrant.who_has(client, skills, "order+gen+cos@100", limit=6):  # 925 = the profile's top implied skill; Qdrant returns ties in id order
    print(f"  {h.payload['label']} ({h.payload['kind']}, {h.score:.0f})")
''')
code(r'''
served = qdrant.served(lambda b: qdrant.who_has(client, [b], "order+gen+cos@100", len(profiles)))
with torch.no_grad(): rows = {"torch, full ranking": down(graph, expanded(graph, order_gen_cos)), "Qdrant, K = 100": down(graph, served)}
stored = load("qdrant.json")
rows["stored run, Qdrant, K = 100"] = stored["order+gen+cos@100"]
table({n: {k: r[k] for k in ("MAP", "R@10", "R@50", "R@100")} for n, r in rows.items()})
''')

md('''
## 13. Prefetch size

The checks above prefetch the whole catalogue, so Qdrant ranks the same candidates as torch. A deployment prefetches fewer. This cell measures what a shorter cosine prefetch costs order + gen + cos, and the time per query (one Qdrant container on a laptop, queries sent one at a time).
''')
code(r'''
rows = {}
for L in (50, 100, 200, ALL):
    ms = []

    def timed(a, L=L):
        t = time.perf_counter()
        hits = order_search(a, *order_w, limit=L, prefetch=L)
        ms.append(1000 * (time.perf_counter() - t))
        return hits

    with torch.no_grad(): r = evaluate(graph, qdrant.served_up(timed), "test", directions=False)
    rows[f"prefetch {L}"] = {"A MAP": r["A"]["MAP"], "B MAP": r["B"]["MAP"], "p50 ms": np.median(ms)}
table(rows)
''')

md('''
## 14. Qdrant against torch

One row per served method. "torch" is the method's own scorer over the full catalogue, "Qdrant" the same method through its collection. The two should agree up to float rounding, since the formulas run in Qdrant and the vectors pass through JSON. The stored columns are the reference runs from `results/` described at the top.
''')
code(r'''
table(PARITY)
gap = max(abs(r[f"{t} torch"] - r[f"{t} Qdrant"]) for r in PARITY.values() for t in "AB")
print(f"largest torch/Qdrant MAP gap: {gap:.4f}")
assert gap < 0.005
''')

md('''
## What this notebook doesn't test

- Scale. Every collection holds 925 skills and the profile collection 1035 profiles. HNSW recall and formula latency at a million points are not measured.
- Concurrency. Latency comes from queries sent one at a time to a local container.
- Qdrant Cloud inference. Section 12 embeds through FastEmbed inside the client; the up sections send the cached encoder rows.
- The box Formula Query beyond one query. It matched the client-side scores on the demo query, and its latency rules it out at this dimension.
- Why the cross-encoder scores lower here than in `results/compare.json`. Both runs use the same training code, seed 0 and three epochs. The stored run also evaluated between epochs, and training on MPS isn't bit-reproducible. The cause of the gap is untested.
- Free-vector ablations, neighbour votes and the Euclidean twin of the hyperbolic model are left out: they are ablations, not methods to serve. They are in `experiments/`.
''')

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
for i, c in enumerate(cells): c["id"] = f"c{i:02d}"
json.dump(nb, open(out, "w"), indent=1)
