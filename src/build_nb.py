"""Builds showcase.ipynb; prose numbers come from results/*.json. Usage: python3 src/build_nb.py showcase.ipynb, then execute it with nbconvert."""
import json, os, sys
out = sys.argv[1]
cells = []
# prose numbers come from the stored results, so a rerun can't leave them stale
res = lambda n: json.load(open(os.path.join(os.path.dirname(os.path.abspath(out)), "results", n + ".json")))
R, D, C, L, Q = map(res, ("recall", "down", "compare", "llm_judge", "qdrant"))
CL, CX, CE = L["claude-sonnet-5"], L["codex-gpt-6-astra"], C["cross_enc rerank50"]
def v(d, k): return f"{d[k]:.3f}"
def pc(x): return f"{x:.0%}"
up = lambda n: v(R[n], "A MAP"); kn = lambda n: v(R[n], "B MAP"); dd = lambda n: v(D[n], "MAP"); dx = lambda n: v(D[n + " (expanded)"], "MAP")
O = "order+gen+cos"
def md(s): cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip()})
def code(s): cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip()})

md("""
# Asymmetric skill matching: a walkthrough

Someone who knows Spring Boot knows Java, because Spring Boot is a Java framework. Someone who knows Java may never have touched Spring Boot. Skill matching has to respect that direction: a recruiter asking for Java should find the Spring Boot developer, and a recruiter asking for Spring Boot should not get every Java developer.

Plain vector search can't do this. Cosine similarity gives the same number in both directions, so it has no way to tell "Spring Boot implies Java" from "Java implies Spring Boot". This notebook compares the ways we tried to fix that, from a one-line heuristic to trained models and LLM rerankers, and runs the best one in Qdrant.

Sections:

1. The problem in one example (live)
2. Words used in this notebook
3. The three tests every method is scored on
4. The strategies, one by one
5. Results for all methods
6. Which options are worth trying, and why
7. Retrain live: a check on the stored numbers
8. What the models predict for skills they have never seen
9. A scenario with CVs and the query "programming in the FP paradigm"
10. Served from Qdrant
11. What isn't tested yet

Sections 1 to 6 are enough for the conclusions. Sections 7 to 10 are evidence and demos. The full write-up is `docs/results.md`.

The labels are provisional. An LLM wrote the skill graph's edges and nobody has audited them yet (`docs/audit_100.csv`). The graph has gaps: it doesn't say that Haskell is a functional programming language, so a model that gets this right is counted as wrong. Read every number as a comparison between methods on the same imperfect labels.

Run the next two cells first. The second one trains the models, which takes about a minute.
""")
code("""
import sys, time; sys.path.insert(0, "src")
from down import *  # data, models, metrics: spike → compare → probe → recall → down
from IPython.display import HTML, Markdown, display

by_label = {n["label"]: ix[q] for q, n in g["nodes"].items()}
def lab(i): return g["nodes"][ids[i]]["label"] if i < len(ids) else f"<new text {i}>"
def show(header, rows):
    display(Markdown("\\n".join(["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
                              + ["| " + " | ".join(map(str, r)) + " |" for r in rows])))
def table(groups, rows):  # groups: [(title, [column, ...]), ...] → a group header row above the column names
    head = "".join(f'<th colspan="{len(cs)}" style="text-align:center">{t}</th>' for t, cs in groups)
    sub = "".join(f"<th>{c}</th>" for _, cs in groups for c in cs)
    body = "".join("<tr>" + "".join(f"<td>{x}</td>" for x in r) + "</tr>" for r in rows)
    display(HTML(f"<table><thead><tr>{head}</tr><tr>{sub}</tr></thead><tbody>{body}</tbody></table>"))
def top(M, q, k=5, pool=tr_list):
    with torch.no_grad(): s = M(torch.tensor([q]), torch.tensor(pool))[0].cpu()
    return [pool[j] for j in s.argsort(descending=True)[:k].tolist()]
def rank_of(m, a, b, pool=tr_list):  # where b lands in a's list of implied skills (1 = top)
    with torch.no_grad(): s = m(torch.tensor([a]), torch.tensor(pool))[0].cpu()
    return int((s > s[pool.index(b)]).sum()) + 1
R, D, C, L, Q = (load(f"results/{f}.json") for f in ("recall", "down", "compare", "llm_judge", "qdrant"))
f = lambda d, k: f"{d[k]:.3f}" if k in d else "–"
print(f"{len(ids)} skills, {len(tr_list)} training skills, {len(g['heldout_nodes']['test'])} held-out test skills, device {dev}")
""")
code("""
M, secs = {"cosine": cos_matrix, "cos+gen": cosgen_matrix(0.05)}, {}
for key in ("dual", "box", "order", "hyp", "transe", "pair_mlp", "folded"):
    t = time.time()
    if key in ("dual", "pair_mlp"): M[key] = trained(key, 0)
    else: M.update(folded(0) if key == "folded" else plus_hybrid(key, trained(key, 0)))
    secs[key] = round(time.time() - t, 1)
print("training seconds per model (seed 0):", secs)
SHORT = ["cosine", "cos+gen", "box+gen+cos", "order+gen+cos", "transe+gen+cos", "pair_mlp"]  # the shortlist from section 6
""")

md("""
## 1. The problem in one example

Each row is a true implication A → B. For each method there are two scores: the true direction s(A, B) and the reverse s(B, A). A method gets the direction right when the true direction scores higher.

Cosine gives the same number both ways, so it can never get the direction right. Order + gen + cos is one of the trained methods from section 4. Two of the four edges were hidden during training, so the model has to infer them from the text.
""")
code("""
PAIRS = [("Spring Boot", "Java"), ("Kotlin", "JVM language"), ("Scala", "functional programming language"), ("WordPress", "PHP")]
m, rows = M["order+gen+cos"], []
with torch.no_grad():
    for a, b in PAIRS:
        A, B = by_label[a], by_label[b]
        fwd, rev = float(m(torch.tensor([A]), torch.tensor([B]))[0, 0]), float(m(torch.tensor([B]), torch.tensor([A]))[0, 0])
        rows.append([f"{a} → {b}", "yes" if B in anc_tr[A] else "no, hidden", f"{float(X[A] @ X[B]):.3f}", f"{float(X[B] @ X[A]):.3f}",
                     f"{fwd:.2f}", f"{rev:.2f}", "✓" if fwd > rev else "✗", rank_of(m, A, B)])
table([("", ["A → B", "edge seen in training?"]), ("cosine", ["s(A, B)", "s(B, A)"]),
       ("order + gen + cos", ["s(A, B)", "s(B, A)", "right direction?", "rank of B among A's 925 implied skills"])], rows)
print(f"Direction accuracy on all true test pairs (ties count half): cosine {R['cosine']['A dir']:.1%}, order + gen + cos {R['order+gen+cos']['A dir']:.1%}")
""")

md("""
## 2. Words used in this notebook

| word | meaning |
|---|---|
| **A** | a skill someone has, usually the more specific one (Spring Boot) |
| **B** | a skill that A implies. When a recruiter asks for it, it's a requirement (Java) |
| **s(A, B)** | a method's score for "A implies B". Higher means more likely |
| catalogue | the 925 skills the models were trained on, with the edges between them |
| held-out skill | one of 110 test skills removed before training. The models only see its text |
| profile | someone who has skills. In the tests a profile is a single skill; in the CV demo it's a CV |
| up | fix A and rank candidate Bs: "what does this person know implicitly?" |
| down | fix B and rank candidate profiles: "who has B, explicitly or implicitly?" |
| expansion | running "up" once per profile at ingest and storing the implied skills |
| generality | log(1 + number of catalogue skills below a skill). "programming language" is general, "Spring Boot" is not |
| MAP | mean average precision, explained below |
| R@K | recall at K: the share of correct answers that land in the top K |
| direction accuracy | for true pairs, the share where s(A, B) > s(B, A), with ties counted as half. 50% is a coin flip, and cosine gets exactly 50% because every pair ties |
| val | the validation split, used to pick settings. Test numbers come from a separate split |

**How MAP works.** Take one query and walk down its ranked list. Each time you reach a correct answer, write down the share of correct answers among the items so far. Average those numbers, then average over all queries. With correct answers at positions 1 and 4, the query scores (1/1 + 2/4) / 2 = 0.75. MAP is 1.0 when every correct answer is ranked above every wrong one. It is the main number in this notebook because it rewards putting correct answers at the very top, which is what a user sees.
""")

md("""
## 3. The three tests

Every method produces one score, s(A, B), and the tests use it in three ways.

**Up, new skill: A → rank B.** A is a held-out skill that no model saw during training. The model reads only its text and ranks all 925 catalogue skills by how likely A implies them. For example, QtJambi was held out, and a good model puts Java and JVM language at the top of its list. This is the closest test we have to a new framework arriving. It says less about CVs: the held-out skills have encyclopedia descriptions like the training skills, and section 9 shows that CV text is harder. The encoder was pretrained on public web text, which may include these descriptions.

**Up, known skill: A → rank new B.** A is a catalogue skill, and the correct answers are implications that were removed from the training graph. FastAPI is in the catalogue, but the edge FastAPI → Python was hidden during training: does Python still come out near the top? This test is about filling gaps in a graph you already have. The column "hit@10, no graph evidence" repeats it on pairs that no graph neighbour of A could hint at, so the model has to infer them from the text.

**Down: B → rank who has it (A).** B is a catalogue skill and plays the requirement. The candidates are 1035 profiles: the 925 catalogue skills plus the 110 held-out ones. Only held-out skills that imply B count as correct. B's own profile and the catalogue skills that imply B are removed from the list, because the graph already connects them to B. For "JVM language", the held-out QtJambi profile should rank near the top. This is the recruiter's search narrowed to its hard part, the profiles the graph can't answer for. There are two ways to answer it:
- *direct*: score every profile against B with s(profile, B) and sort;
- *via expansion*: run the up direction once per profile at ingest time and store its implied skills. The search then sorts profiles by how high B sits in each profile's list. Section 10 runs this route in Qdrant.

Direction matters in all three. For "who knows Java?", a Spring Boot profile is a correct answer and an "object-oriented programming" profile is a wrong one: someone who knows OOP might know C++ and no Java at all. The case where the recruiter's requirement isn't a catalogue skill at all is not one of the three tests (section 9 shows it once).
""")

md("""
## 4. The strategies, one by one

The trained methods all start from the same input: a 768-number text vector from the bge-base encoder, computed from each skill's label and description. The encoder stays frozen. Each method trains a small two-layer network on top of it. Most of them train on the same data:
- *positives* are pairs (A, B) where A implies B in the training graph, including chains (Spring Boot → Java → JVM language gives Spring Boot → JVM language);
- *negatives* are the reversed pair (B, A), a random unrelated skill, and a skill that shares the parent B but isn't implied by A.

Two methods differ. Hyperbolic embeddings contrast each positive with 10 random skills that are neither above nor below A in the graph, in a softmax loss, as in the Nickel and Kiela paper they come from. The distilled dual trains to copy another model's rankings instead of learning from the pairs directly.

### 4.1 No training

**Cosine.** The cosine of the angle between the two text vectors: s(A, B) = cos(x_A, x_B). It finds related skills well, but s(A, B) = s(B, A) always, so its direction accuracy is exactly 50%. It is the plain vector search baseline.

**Popularity.** s(A, B) = generality(B). It ignores A and ranks general skills first, so on its own it ranks badly. It is in the table because requirements tend to be general skills, and that prior helps the other methods.

**Cosine + generality (cos + gen).** s(A, B) = cos(x_A, x_B) + λ·generality(B), with λ = 0.05. Read it as "similar, and more general". It is asymmetric because only B's generality counts: for Spring Boot → Java the large generality of Java is added, and for Java → Spring Boot only the small generality of Spring Boot. It needs no training and no labels beyond the catalogue's edge counts. The rerankers work on its top 50.

### 4.2 Trained models that read the text vector

**Dual encoder.** Two networks, one for the "has" role and one for the "required" role: s(A, B) = has(x_A) · req(x_B). It is asymmetric because the two roles use different vectors. It runs as a single dot product, which Qdrant does natively with named vectors.

**Box embeddings.** Each skill becomes a box, a region of space, and general skills learn big boxes around the boxes of their specific skills. s(A, B) = log P(B | A) = log(volume of A ∩ B) − log(volume of A), the share of A's box that lies inside B's box. It is asymmetric because containment is: Spring Boot's box sits inside Java's box, and Java's box sticks out of Spring Boot's. The score is a log probability (smoothed, so that boxes that don't overlap still get a gradient), which fits the graded labels we plan to use, and intersecting boxes gives a natural score for a profile with several skills. Neither use is tested yet.

**Order embeddings.** Each skill becomes a point with non-negative coordinates, and specific skills sit further from the origin. A implies B when A is at least as large as B on every coordinate. The score is the penalty for every coordinate where B exceeds A, s(A, B) = −Σᵢ max(0, Bᵢ − Aᵢ)², which is 0 when A covers B fully. Think of the coordinates as ingredients: Spring Boot has at least as much of every ingredient as Java, plus some of its own. It is asymmetric because "covers" is.

**Hyperbolic embeddings.** Skills become points in a curved space, the Poincaré ball, which has far more room near its edge than near its centre. That suits trees: general skills go near the centre, specific ones near the edge. This follows the Qdrant article on hyperbolic embeddings. Its score is minus the distance between the two points, and distance is symmetric, so plain hyperbolic has 50% direction accuracy. The "is-a" variant multiplies the distance by 1 + α(‖B‖ − ‖A‖), which shrinks it when B is nearer the centre than A. That fixes the direction (97%) and leaves the ranking where it was.

**TransE.** The model learns one "implies" vector r and places skills so that f(A) + r lands near f(B): s(A, B) = −‖f(A) + r − f(B)‖. It is asymmetric because r points one way. TransE was built for graphs with several relation types, one vector per type. Our graph has one relation, so that strength goes unused.

**Pairwise classifier (pair_mlp).** A network reads both vectors at once, [x_A, x_B, x_A ⊙ x_B, x_A − x_B], and outputs a logit: the number a sigmoid would turn into P(A implies B). It is asymmetric because swapping the inputs changes the answer. It can't go in a vector index: every (A, B) pair needs its own pass through the network.

### 4.3 The hybrid recipe: "+ gen + cos"

A trained score plus the two free signals: s(A, B) = model(A, B) + λ·generality(B) + μ·cos(x_A, x_B), with λ and μ picked on val for each model. The trained models learn from fewer than a thousand edges, and on their own they forget some of what the encoder knows. Adding cosine brings back general text similarity, and generality adds the prior that requirements are general skills. Every trained model gains from it. The sum is no longer a probability, even when the model part is one.

**Distilled dual, folded.** A dual encoder trained to copy the rankings of box + gen + cos, with generality and cosine folded into extra dimensions so that the whole hybrid is still one dot product. It was built for query-time search in a single HNSW index.

### 4.4 Rerankers

A reranker scores only a shortlist, here the top 50 from cos + gen. That shortlist holds 99% of the correct answers for "up, new skill" and 93% for "up, known skill", so reranking loses little. The 50 are then sorted by cos + gen plus the reranker's score, and they stay above the rest of the catalogue. Sorting by the LLM's score alone, with cos + gen only breaking ties, did worse on 3 of the 4 LLM numbers (Claude 0.779 and 0.681, Codex 0.726 and 0.630, against 0.773, 0.698, 0.757 and 0.686 for the sum), so the sum stays.

**Cross-encoder.** A transformer (bge-reranker-base) reads the two texts together and outputs a score, which a sigmoid turns into a probability. We fine-tuned it on the training pairs. It is asymmetric because the order of the texts matters. Retrieve-then-rerank with a cross-encoder is what most teams would build, so it is the industry baseline here. It had one seed and three epochs, and it was still improving when training stopped.

**LLM judge.** Claude Sonnet 5 and Codex (gpt-6-astra) get the 50 candidates and score each one as p(A → B): "the probability that a person with solid working experience in A has at least basic working competence in B". There is no training. One call covers 5 queries and takes about 45 s for Claude and 75 s for Codex, so roughly 9 s and 15 s of LLM time per query. The LLM is the best reference we have, and every cheaper method should be read against it. Two limits apply. Its recall is capped by the cos + gen top 50. And Claude wrote the labels, which could favour a Claude judge; Codex, from another model family, scores close to it.

### 4.5 Methods that only work inside the catalogue

Neighbour vote, vote + popularity, and free vectors (one learned vector per catalogue skill with no text, as in the hyperbolic article) need the skill to be a node in the training graph. They have no score for new text, so they only take part in "up, known skill". They are in the table to show how much of that test the graph alone can solve.

### 4.6 Direct scoring or expansion for the down search

Any method can answer "who has B?" directly by sorting profiles on s(profile, B). The alternative is expansion: at ingest, each profile ranks the 925 catalogue skills in the up direction and stores its top K. The search then returns the profiles whose stored list contains B, best rank first. Expansion only uses the up direction, which is the one the models were trained and tuned on, and it turns the search into a sparse-vector lookup. Its limit is that it can only store catalogue skills. Both routes use the settings (λ, μ) tuned for the up direction; direct scoring tuned for the down search could narrow the gap, and that is untested.
""")

md("""
## 5. Results for all methods

Test split, mean of 3 seeds, loaded from `results/*.json`. The first table is the shortlist from section 6; the second has every method.
- *Up, new skill* and *up, known skill* are the tests from section 3. R@50 caps what a reranker working on the top 50 can reach.
- *Down* is the recruiter's search, direct and via expansion.
- *Train (s)* is seconds per training run on an M5 Pro, including the λ, μ tuning.
- *Serving* is how the method would run in Qdrant. Only the via-expansion route of order + gen + cos and cos + gen has run there (section 10).

Ties in a ranking are broken at random, the same way for every method. A dash means the method can't run that test: graph-only methods have no score for new text, and the rerankers weren't run in the down direction because the cached LLM answers only cover up-direction queries.
""")
code("""
UP_NEW, UP_KNOWN, DOWN = "Up, new skill: A → rank B", "Up, known skill: A → rank new B", "Down: B → rank who has it (A)"
ROLE = {"cosine": "floor", "cos+gen": "free baseline", "box+gen+cos": "second pick", "order+gen+cos": "first pick",
        "transe+gen+cos": "keep only with more relation types", "pair_mlp": "drop"}
QCOST = {"cosine": "1 vector search", "cos+gen": "1 vector search + a stored count",
         "box+gen+cos": "sparse lookup after expansion (not run)", "order+gen+cos": f"sparse lookup after expansion: {Q['p50 ms (link + top 10)']:.1f} ms p50 in Qdrant",
         "transe+gen+cos": "sparse lookup after expansion (not run)", "pair_mlp": "rerank, or expansion (not run)"}
pct = lambda d, k: f"{d[k]:.0%}" if k in d else "–"
def row(n): return [n, ROLE[n], f(R[n], "A MAP"), f(R[n], "B MAP"), pct(R[n], "A dir"), f(D[n], "MAP"), f(D[n + " (expanded)"], "MAP"),
                    f"{R[n]['train s']:.0f}", QCOST[n]]
RERANKERS = [("cross-encoder, reranks cos+gen top 50", "industry baseline", C["cross_enc rerank50"], "fine-tuned", "50 transformer passes"),
             ("LLM Claude Sonnet 5, reranks top 50", "best reference", L["claude-sonnet-5"], "none", "about 9 s of LLM time"),
             ("LLM Codex, reranks top 50", "best reference", L["codex-gpt-6-astra"], "none", "about 15 s of LLM time")]
rows = [row(n) for n in SHORT[:2]] + [[n, role, f(r["A"], "MAP"), f(r["B"], "MAP"), pct(r["A"], "dir"), "–", "–", tr, qc]
                                      for n, role, r, tr, qc in RERANKERS] + [row(n) for n in SHORT[2:]]
display(Markdown("**Shortlist**"))
table([("", ["method", "verdict"]), ("Up (MAP)", ["new skill", "known skill"]), ("", ["direction accuracy"]),
       ("Down (MAP)", ["direct", "via expansion"]), ("Cost", ["train (s)", "per query"])], rows)
""")
code("""
SERVE = {  # how it would run in Qdrant; only the two "tested" entries have run
    "cosine": "1 vector, HNSW cosine", "cos+gen": "HNSW + payload boost; expansion → sparse vector (tested)", "pop": "payload only",
    "vote": "graph lookup (catalogue only)", "vote+pop": "graph lookup (catalogue only)", "hyp (free vectors)": "catalogue only, rerank formula",
    "dual": "2 named vectors, HNSW dot", "distilled dual, folded": "2 named vectors, HNSW dot",
    "box": "rerank formula or range filters", "box+gen+cos": "prefetch + rerank formula",
    "order": "rerank formula or range filters", "order+gen+cos": "prefetch + rerank formula; expansion → sparse vector (tested)",
    "hyp": "rerank formula (article)", "hyp+gen+cos": "prefetch + rerank formula", "hyp+isa": "rerank formula",
    "transe": "HNSW Manhattan on f(a)+r", "transe+gen+cos": "HNSW + rerank formula", "pair_mlp": "rerank only"}
rows = [[n, f(r, "A MAP"), f(r, "A R@10"), pct(r, "A dir"), f(r, "B MAP"), f(r, "B R@50"), f(r, "no nb"), f(D.get(n, {}), "MAP"),
         f(D.get(n + " (expanded)", {}), "MAP"), f"{r['train s']:.0f}", SERVE[n]] for n, r in R.items()]
for n, r in [("cross-encoder, rerank top 50", C["cross_enc rerank50"]), ("LLM Claude Sonnet 5, rerank top 50", L["claude-sonnet-5"]),
             ("LLM Codex, rerank top 50", L["codex-gpt-6-astra"])]:
    rows.append([n, f(r["A"], "MAP"), f(r["A"], "R@10"), pct(r["A"], "dir"), f(r["B"], "MAP"), "–", "–", "–", "–", "–", "rerank only"])
display(Markdown("**All methods**"))
table([("", ["method"]), (UP_NEW, ["MAP", "R@10", "direction"]), (UP_KNOWN, ["MAP", "R@50", "hit@10, no graph evidence"]),
       (DOWN, ["MAP direct", "MAP via expansion"]), ("Cost", ["train (s)", "serving"])], rows)
""")
F = "distilled dual, folded"
direct = [n for n in D if not n.endswith("(expanded)")]
assert max(direct, key=lambda n: D[n]["MAP"]) == O and max(direct, key=lambda n: D[n + " (expanded)"]["MAP"]) == O
assert all(D[n + " (expanded)"]["MAP"] > D[n]["MAP"] for n in direct)
assert D["cos+gen (expanded)"]["MAP"] > max(D[n]["MAP"] for n in direct)
assert {n for n in direct if D[n]["MAP"] > D["cosine"]["MAP"]} == {"box+gen+cos", O, "transe+gen+cos"}
md(f"""
Three things stand out beyond the shortlist.

The down search is much harder than the up direction. The best direct score is {dd(O)} (order + gen + cos) against {dd("cosine")} for cosine, while the same model scores {up(O)} on "up, new skill". Only the three + gen + cos hybrids of box, order and TransE beat cosine at direct scoring.

Expansion beats direct scoring for every model. Cos + gen via expansion reaches {dx("cos+gen")} with no training, above every model's direct score, and order + gen + cos reaches {dx(O)}.

The hyperbolic models, TransE and the two duals do worst at direct scoring: {dd("hyp")} for plain hyperbolic, {dd("hyp+gen+cos")} for hyperbolic + gen + cos, {dd("transe")} for TransE, {dd("dual")} for the dual and {dd(F)} for the folded dual. Via expansion, hyperbolic + gen + cos recovers to {dx("hyp+gen+cos")} and the folded dual to {dx(F)}, while TransE and the dual stay near 0.1 ({dx("transe")} and {dx("dual")}).
""")

assert max(R, key=lambda n: R[n].get("A MAP", 0)) == O
B_ = "box+gen+cos"; T_ = "transe+gen+cos"; P_ = "pair_mlp"; CG = "cos+gen"; H_ = "hyp+gen+cos"
ms = f"{Q['p50 ms (link + top 10)']:.1f} ms"
md(f"""
## 6. Which options are worth trying, and why

| option | role | verdict |
|---|---|---|
| cosine | floor | keep as the reference point |
| cos + gen | free baseline | keep: every trained method has to beat it |
| cross-encoder reranking the top 50 | industry baseline | keep as the comparison most teams expect |
| LLM reranking the top 50 | best reference | keep as the comparison branch |
| order + gen + cos | candidate | first pick |
| box + gen + cos | candidate | second pick, with the most untested upside |
| TransE + gen + cos | candidate | keep only if the graph gets more relation types |
| pairwise MLP | candidate | drop, unless pair_mlp + gen + cos changes the picture |
| dual, distilled dual, hyperbolic, graph-only methods, popularity | | not worth more time now |

### Baselines

**Cosine** is what a plain vector search gives you: {up("cosine")} on "up, new skill", {dd("cosine")} on the down search, and 50% direction accuracy.

**Cos + gen** doubles cosine on "up, new skill" ({up(CG)}) with one added term and no training. Via expansion it reaches {dx(CG)} on the down search, and the rerankers work on its top 50. It is the honest "what you get for free" baseline: a trained model that can't clearly beat it isn't worth deploying.

**Cross-encoder reranking** is the standard retrieve-then-rerank setup, at {v(CE["A"], "MAP")} on up-new and {v(CE["B"], "MAP")} on up-known. Order + gen + cos scores above it on both ({up(O)} and {kn(O)}) without a transformer pass per candidate at query time. The comparison is weak, though: the cross-encoder had one seed and three epochs and was still improving, so the gap could shrink or close.

**LLM reranking** is the best reference we have. Claude reaches {v(CL["A"], "MAP")} on up-new and {v(CL["B"], "MAP")} on up-known, and in the CV demo (section 9) it is the only method the recruiter's CV doesn't fool. The price is about 9 s of LLM time per query. Order + gen + cos reaches {pc(R[O]["A MAP"] / CL["A"]["MAP"])} of Claude's MAP on up-new and {pc(R[O]["B MAP"] / CL["B"]["MAP"])} on up-known, with no LLM call. Speed is a separate measurement: the down search with order + gen + cos takes {ms} at the median in Qdrant, on the 1035-profile test collection. The LLM wasn't run on the down search, so there is no quality comparison there. The LLM is better, and that should be said plainly. Two untested routes could close the gap: the LLM as a teacher (graded labels, then train order or box on them), and the LLM expanding each CV once at ingest.

### Candidates

**Order + gen + cos is the first pick.** It has the best trained score on up-new ({up(O)}, with {pc(R[O]["A R@10"])} of correct answers in its top 10) and the best down search both ways ({dd(O)} direct, {dx(O)} via expansion). It trains in about 3 s and already runs in Qdrant (section 10).

**Box + gen + cos is the second pick**, and the one with the most upside we haven't measured. It is ahead of order on up-known ({kn(B_)} against {kn(O)}) and on pairs with no graph evidence (hit@10 {v(R[B_], "no nb")} against {v(R[O], "no nb")}). It is behind on up-new ({up(B_)}) and on the down search ({dx(B_)} via expansion). What box has and order lacks is a score with a meaning: its box part is log P(B | A) before gen and cos are added. That matches the graded labels we plan to collect, and box intersections give a natural score for a CV with several skills. Both are untested.

**TransE + gen + cos** is close to box + gen + cos: {up(T_)} on up-new, {kn(T_)} on up-known, {dx(T_)} via expansion. Most of that comes from the gen + cos part, though, because TransE alone scores {up("transe")} on up-new, below cosine. Its one distinctive feature is the relation vector, which pays off in a graph with several relation types, such as "requires", "is an alternative to" and "is used with". With the single relation we have, order and box do the same job better.

**The pairwise MLP I'd drop.** It ties cos + gen on up-new ({up(P_)} against {up(CG)}) and loses to it on up-known ({kn(P_)} against {kn(CG)}), on pairs with no graph evidence ({v(R[P_], "no nb")} against {v(R[CG], "no nb")}) and on the down search via expansion ({dx(P_)} against {dx(CG)}). It needs training and still doesn't beat the free baseline. Its serving limit (rerank only) matters less once profiles are expanded offline, so the quality numbers are the real reason. One run could change this verdict: pair_mlp + gen + cos was never tried.

### Not worth more time now

- Dual and distilled dual. Their selling point is a single dot product in HNSW, which expansion at ingest makes unnecessary, and both are weaker: the dual scores {up("dual")} on up-new, and the folded dual {up(F)} on up-new and {dd(F)} on the direct down search.
- Hyperbolic. The article's distance is symmetric, so direction accuracy is 50%, and it stays 50% with gen + cos because val gave generality no weight. It has the best deep recall on up-known (R@50 {v(R[H_], "B R@50")}), but deep recall is a first-stage property, and cos + gen already holds {pc(R[CG]["B R@50"])} there.
- Graph-only methods (vote, free vectors). They have no score for new text, so they can't handle a CV or a new framework.
- Popularity. It can score new text, but it ignores A, so every skill gets the same list of implied skills, and it reaches only {up("pop")} on up-new.
""")

md("""
## 7. Retrain live: a check on the stored numbers

The models trained at the top of the notebook use seed 0, with configs from the val sweeps (`results/sweep_best.json`, `results/compare_sweep.json`). Expect numbers a few hundredths off the stored ones: this is one seed, and results on the Apple GPU vary slightly between runs.
""")
code("""
SHOW = ["cosine", "cos+gen", "dual", "distilled dual, folded", "box+gen+cos", "order+gen+cos", "hyp+gen+cos", "transe+gen+cos", "pair_mlp"]
with torch.no_grad(): live = {n: (evaluate(M[n], "test"), down(M[n]), down(expanded(M[n]))) for n in SHOW}
table([("", ["model"]), (UP_NEW, ["MAP live", "MAP 3 seeds"]), (UP_KNOWN, ["MAP live", "MAP 3 seeds"]),
       (DOWN, ["MAP direct, live", "MAP direct, 3 seeds", "MAP via expansion, live", "MAP via expansion, 3 seeds"])],
      [[n, f(e["A"], "MAP"), f(R[n], "A MAP"), f(e["B"], "MAP"), f(R[n], "B MAP"), f(d, "MAP"), f(D[n], "MAP"),
        f(x, "MAP"), f(D[n + " (expanded)"], "MAP")] for n, (e, d, x) in live.items()])
""")

md("""
## 8. What the models predict for skills they have never seen

These four are held-out test skills. Each shortlisted method ranks the 925 catalogue skills, and the table shows its top 5. A ✓ marks a skill the graph lists as a true requirement. The LLM row is Claude Sonnet 5 reranking the cos + gen top 50. Its answers are cached, so this cell makes no LLM call.
""")
code("""
from llm_judge import judge_all, Judged
llm = Judged(*judge_all("claude-sonnet-5")[:2])
for name in ["Express.js", "Amazon DynamoDB", "Thymeleaf", "AUCTeX"]:
    q = by_label[name]; truth = anc_full[q] & set(tr_list)
    display(Markdown(f"**{name}**. True requirements: {', '.join(sorted(lab(b) for b in truth))}"))
    show(["method", "top 5 implied skills"], [[n, ", ".join(lab(b) + (" ✓" if b in truth else "") for b in top(m, q))]
                                             for n, m in {**{n: M[n] for n in SHORT}, "LLM Claude (cached)": llm}.items()])
""")

md("""
## 9. A scenario with CVs and the query "programming in the FP paradigm"

Six short CVs, all new text that no model has seen. My labels say whether each person programs in a functional style. The recruiter CV is a trap: it names FP languages, and the person doesn't program in any of them.

The query can be answered the two ways from section 3. Via expansion, each CV ranks the catalogue skills it implies, and the CVs whose list contains FP match (9b). Directly, each CV is scored against the requirement (9c).

It also matters whether FP exists as a catalogue skill.
- Case 1, FP is in the catalogue: link the query text to the closest catalogue skill by cosine. Matching paraphrases is what cosine is good at.
- Case 2, FP isn't in the catalogue: encode the query text itself in the requirement role. This is untested, and expansion can't do it, because it can only store catalogue skills.

Six CVs make an anecdote. The measurements are in section 5.
""")
code("""
CVS = {  # name: (CV overview text, programs in FP? my label)
    "Scala engineer":   ("Senior software engineer, six years of Scala. Built backend services and batch data pipelines.", True),
    "Haskell dev":      ("Haskell developer writing compilers and type-level domain-specific languages.", True),
    "Elixir dev":       ("Elixir and Phoenix developer building real-time web applications.", True),
    "Java/Spring dev":  ("Java developer building Spring Boot microservices with REST APIs and PostgreSQL.", False),
    "Python DS":        ("Data scientist using Python, pandas and scikit-learn for forecasting models.", False),
    "Recruiter (trap)": ("Technical recruiter hiring functional programming engineers for Scala, Haskell and Clojure roles.", False),
}
QUERY = "Programming in the functional programming paradigm"
cv = dict(zip(CVS, add_texts([t for t, _ in CVS.values()])))
qrow, = add_texts([QUERY])
link = top(cos_matrix, qrow, k=5)
show(["rank", "catalogue skill closest to the query (cosine)", "cos"], [[k + 1, lab(b), f"{float(X[qrow] @ X[b]):.3f}"] for k, b in enumerate(link)])
fp = link[0]
print("case 1 requirement:", lab(fp))
""")
md("### 9a. What the Scala CV implies (up direction, top 8 of 925)")
code("""
show(["method", "top 8 implied skills"], [[n, ", ".join(lab(b) for b in top(M[n], cv["Scala engineer"], k=8))] for n in SHORT])
""")
md("""
### 9b. Via expansion: where FP ranks in each CV's implied skills

Each cell is FP's rank among the 925 catalogue skills for that CV (1 = top), and a ✓ marks the CVs that should match. If each profile stores its top K implied skills, a CV matches the query when FP's rank is K or better.
""")
code("""
show(["method"] + [c + (" ✓" if CVS[c][1] else "") for c in CVS], [[n] + [rank_of(M[n], cv[c], fp) for c in CVS] for n in SHORT])
""")
md("""
### 9c. Direct: rank the six CVs for the requirement

P@3 is the share of the top 3 who program in FP (3 of the 6 do). Case 1 scores the CVs against the FP catalogue skill, case 2 against the query text used as a requirement.
""")
code("""
names = list(cv)
def who(m, b):
    with torch.no_grad(): s = m(torch.tensor(list(cv.values())), torch.tensor([b]))[:, 0].cpu()
    order = [names[j] for j in s.argsort(descending=True).tolist()]
    return f"{sum(CVS[c][1] for c in order[:3]) / 3:.2f}", " > ".join(c + (" ✓" if CVS[c][1] else "") for c in order)
show(["method", "case 1 P@3", "case 1: CVs ranked for the FP skill", "case 2 P@3", "case 2: CVs ranked for the query text"],
     [[n, *who(M[n], fp), *who(M[n], qrow)] for n in SHORT])
""")
md("""
### 9d. The LLM on the same CVs

The LLM needs no catalogue, so it covers both cases. The first run makes one live `claude -p` call (about 20 s), and later runs read the answer from `data/llm_cache/`. The question is the same p(A → B) the LLM judge answers, with A = the CV text and B = the query.
""")
code("""
from llm_judge import ask, DEF
prompt = "\\n".join([DEF, "", 'Score p(A→B) for every numbered pair. Answer with one JSON object only: {"1": p, "2": p, ...}, '
                     "p in [0, 1], two decimals.", ""] + [f"{j}. A = {t} | B = {QUERY}" for j, (t, _) in enumerate(CVS.values(), 1)])
p = ask("claude-sonnet-5", prompt) or {}
show(["CV", "programs in FP (my label)", "LLM p(CV → query)"], [[c, "✓" if CVS[c][1] else "", p.get(str(j), "–")] for j, c in enumerate(CVS, 1)])
""")
md("""
**What the scenario shows.** The query text links to "functional programming language", so case 1 starts from the right skill. The box and order expansions of the Scala CV contain Scala, JVM language and FP, along with general languages such as Python and ruby that the generality term pushes up. TransE's top 8 for the same CV is unrelated tooling. FP ranks high for the non-FP CVs too. Every embedding method falls for the recruiter trap, because none of them can tell "mentions FP" from "does FP". The models were trained on encyclopedia descriptions of skills, and CV text is new ground for them. The LLM separates the two groups, trap included.
""")

md("""
## 10. Served from Qdrant

The code is in `src/qdrant_serve.py`. It needs a Qdrant server on localhost:6333: `podman run -d -p 6333:6333 docker.io/qdrant/qdrant`.

At ingest, Qdrant inference (`models.Document`) turns each text into a dense `text` vector. Here FastEmbed runs it inside the Python client; on Qdrant Cloud the same call runs on the server. A trained method then expands each profile into its top K implied catalogue skills and stores them as a sparse vector named `<method>@<K>`, with index = skill id and value = 926 − rank, so the top skill has value 925. There are two collections: `skills` holds the 925 catalogue skills, and `profiles` holds the 1035 profiles from the down test plus the 6 CVs.

At query time, the requirement text goes to a dense search in `skills`, which returns a skill id. A sparse query `{skill id: 1}` on `profiles` then returns the profiles whose expansion contains that skill, best rank first. Several requirements become several indices, and their values add up. Payload filters work as usual: `kind = cv` keeps only the CVs.

The methods are the live seed-0 models from the top of the notebook. Ingest runs FastEmbed on about 2000 texts on the CPU, so it takes a bit over a minute.
""")
code("""
import qdrant_serve as qs
from qdrant_client import models
heads = {"order+gen+cos": M["order+gen+cos"], "cos+gen": M["cos+gen"]}
t0 = time.time()
with torch.no_grad():
    qs.build(heads)
    qs.ingest(heads, list(cv.values()), [txt for txt, _ in CVS.values()], [{"label": c, "kind": "cv"} for c in CVS])
print(f"{qs.qc.count('skills').count} skills, {qs.qc.count('profiles').count} profiles (1035 skill profiles + 6 CVs), {time.time() - t0:.0f}s")
""")
md("### 10a. The FP query, end to end in Qdrant")
code("""
hit = qs.link(QUERY, 3)
print("query → skills (dense search):", ", ".join(lab(b) for b in hit))
only_cvs = models.Filter(must=[models.FieldCondition(key="kind", match=models.MatchValue(value="cv"))])
for using in ("order+gen+cos@100", "cos+gen@100"):
    pts = qs.who_has(hit[:1], using, 6, query_filter=only_cvs)
    missing = [c for c in CVS if c not in {p.payload["label"] for p in pts}]
    display(Markdown(f"**{using}**, CVs only. FP's rank is its place in the CV's stored expansion (1 = top); CVs past K = 100 aren't returned."))
    show(["result", "CV", "programs in FP (my label)", "FP's rank"], [[k + 1, p.payload["label"], "✓" if CVS[p.payload["label"]][1] else "",
                                                              926 - int(p.score)] for k, p in enumerate(pts)]
         + [["–", c, "✓" if CVS[c][1] else "", "> 100"] for c in missing])
""")
md("### 10b. Two requirements in one query: FP and JVM language")
code("""
both = [fp, by_label["JVM language"]]
pts = qs.who_has(both, "order+gen+cos@100", 8)
show(["result", "profile", "kind", "score (sum of 926 − rank)"], [[k + 1, p.payload["label"], p.payload["kind"], int(p.score)] for k, p in enumerate(pts)])
""")
md("""
### 10c. Measured: the down test served from Qdrant (`results/qdrant.json`, seed 0)

These are the same 69 down-test queries as in section 5, run through Qdrant.
- *torch* is the same ranking computed in memory.
- *@K* stores K implied skills per profile.
- *label text, linked* starts from the requirement's label as free text, links it to a skill with a dense search, then runs the sparse lookup.
- *dense, direct* is plain cosine over the profiles' `text` vectors.
""")
code("""
show(["route", "down MAP", "R@10", "R@50", "R@100"], [[n, *(f"{Q[n][k]:.3f}" for k in ("MAP", "R@10", "R@50", "R@100"))] for n in Q if isinstance(Q[n], dict)])
print(f"link@1 (label text → right skill): {Q['link@1']:.3f}; p50 latency, link + top 10: {Q['p50 ms (link + top 10)']:.1f} ms; "
      f"ingest {Q['ingest s']:.0f} s")
""")
q = lambda k, m="MAP": Q[f"{O}{k}"][m]
assert abs(q("@925") - q(", torch")) < 1e-9 and abs(q("@100") - q(", torch")) < 0.005
md(f"""
Storing all 925 implied skills gives the in-memory ranking back: Qdrant's scores differ from the torch ones by a constant, so under the same random tie-breaking the metrics match (MAP {q("@925"):.3f} both ways). Storing 100 per profile keeps MAP at {q("@100"):.3f}, but R@50 drops from {q(", torch", "R@50"):.3f} to {q("@100", "R@50"):.3f} and R@100 from {q(", torch", "R@100"):.3f} to {q("@100", "R@100"):.3f}. Profiles whose list doesn't have the skill in its top 100 aren't returned; they tie last and the metrics order them at random, which still gives them a little credit. Storing 10 per profile drops MAP to {q("@10"):.3f}. Starting from free label text costs {q(", torch") - q("@100, label text linked"):.2f}, because dense search links a label to its own skill {pc(Q["link@1"])} of the time. The median query takes {ms} for embedding, linking and a top-10 lookup, measured on the 1035-profile test collection.

Qdrant also shows the scenario's weakness (10a). FP is in the top 10 of every CV's order + gen + cos expansion, so K = 100 returns all six CVs, and the recruiter ties the Haskell developer at rank 1.
""")

md(f"""
## 11. What isn't tested yet

- Real CVs. The six CVs are an anecdote, and CV text is where the embedding methods fail. A set of real CVs with judged skills would settle it.
- A direct route tuned for the down search. Both routes use the (λ, μ) tuned for the up direction.
- The LLM at ingest. Letting the LLM expand each CV once, instead of reranking per query, could give its quality at the {ms} search cost.
- The LLM as a teacher: graded labels from the LLM, then order or box trained on them. It needs your go.
- Requirements outside the catalogue (case 2 in section 9), as a measured split.
- Profiles with several skills: box intersection against order's coordinate-wise maximum.
- Qdrant at scale and Qdrant Cloud inference. Only 1041 points ran here, through the local FastEmbed path.
- pair_mlp + gen + cos, the one run that could change the verdict on the pairwise classifier.
- The label audit (`docs/audit_100.csv`). Every number above depends on it.
""")
nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
for i, c in enumerate(cells): c["id"] = f"c{i:02d}"
json.dump(nb, open(out, "w"), indent=1)
