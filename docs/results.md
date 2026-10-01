# Asymmetric skill implication — model comparison (round 3)

Data, splits, negatives and metrics as in `skillmatch/` (`data.py`, `training.py`, `metrics.py`, `tasks.py`): 1,106 skills, 1,253 training pairs (graph closure).
Test sets: 99 held-out skills with a training ancestor (Task A: rank training skills as ancestors of an unseen skill) and 183 pairs implied by 104 held-out
edges (Task B: filtered ranking over 114 query skills). MAP is the main score; `dir` = share of true pairs scored higher in
the true direction (0.5 = coin flip). Learned models: 3-seed means (MAP sd ≤ 0.03), configs picked on val only.

**Val / test overlap (fixed 2026-09-28).** 4 of the 65 val pairs were also test pairs (implied by both a val and a test edge). `spike.py` now drops them from val, so they count in test only. `recall.py`, `down.py`, `qdrant_serve.py` and `compare.py test` were rerun with the filter. Only order + gen + cos moved because of it (val picks different (λ, μ) for some seeds; 0.622 / 0.546 → 0.618 / 0.529). The `compare.py test` rerun also picked up the earlier tie-break fix (order B 0.479 → 0.464) and cross-process MPS noise in the free-vector models (free hyperbolic 0.338 → 0.317, + is-a 0.314 → 0.341, order with N&K loss 0.164 → 0.105; see the caveat under the free-vector section). `spike.py` was rerun as well: only box moved (A 0.485 → 0.478, B 0.500 → 0.501). Still predating both fixes: the distillation and cross-encoder rows, and the val sweeps with the configs they picked.

## Results (test)

| model | A: MAP | A: R@10 | A: dir | B: MAP | B: R@10 | B: dir | fit MAP | Qdrant serving |
|---|---|---|---|---|---|---|---|---|
| cosine | 0.271 | 0.459 | 0.500 | 0.183 | 0.307 | 0.500 | – | native |
| cosine + generality | 0.520 | 0.770 | 0.984 | 0.464 | 0.760 | 0.951 | – | native + formula over a payload count |
| dual (two roles) | 0.371 | 0.679 | 0.988 | 0.355 | 0.645 | 0.962 | – | two named vectors, native HNSW |
| box | 0.478 | 0.767 | 0.977 | 0.501 | 0.728 | 0.964 | – | prefetch + rerank; containment as range filters |
| box + generality + cosine | 0.582 | 0.815 | 0.979 | 0.540 | 0.765 | 0.964 | – | prefetch + rerank |
| order | 0.472 | 0.794 | 0.984 | 0.464 | 0.730 | 0.967 | 0.757 | prefetch + rerank; dominance as range filters |
| **order + generality + cosine** | **0.618** | **0.878** | 0.986 | 0.529 | **0.782** | 0.967 | – | prefetch + rerank |
| TransE (h + r) | 0.214 | 0.424 | 0.961 | 0.356 | 0.597 | 0.951 | 0.736 | native HNSW: query vector f(a) + r, Manhattan |
| TransE + generality + cosine | 0.576 | 0.803 | 0.966 | 0.524 | 0.767 | 0.954 | – | prefetch + rerank |
| distilled dual (teacher: box + gen + cos) | 0.465 | 0.755 | 0.988 | 0.532 | 0.756 | 0.978 | 0.745 | two named vectors, native HNSW |
| distilled dual, gen + cos folded in | 0.496 | 0.720 | 0.984 | 0.531 | 0.765 | 0.973 | 0.794 | one dot product, native HNSW |
| hyperbolic (article, symmetric) | 0.424 | 0.605 | 0.500 | 0.459 | 0.661 | 0.500 | 0.885 | article pipeline: Euclid prefetch + acosh formula |
| hyperbolic + is-a score | 0.422 | 0.605 | 0.972 | 0.460 | 0.661 | 0.974 | 0.884 | same + one norm term from payload `sq_norm` |
| hyperbolic + generality + cosine (val picks λ = 0, so still symmetric) | 0.547 | 0.779 | 0.500 | **0.544** | 0.768 | 0.500 | – | article pipeline + cosine in the formula |
| flat twin of hyperbolic (Euclidean, 128 dims) | 0.264 | 0.532 | 0.500 | 0.156 | 0.345 | 0.500 | 0.683 | native |
| pairwise MLP | 0.521 | 0.806 | 0.989 | 0.411 | 0.634 | 0.969 | 0.499 | rerank only |
| pairwise MLP + cosine (val picks λ = 0) | 0.578 | 0.842 | 0.989 | 0.444 | 0.694 | 0.969 | – | rerank only |
| cross-encoder (full ranking) | 0.514 | 0.782 | 0.979 | 0.421 | 0.652 | 0.984 | – | rerank only |
| cross-encoder, rerank top-50 | 0.601 | 0.820 | 0.979 | 0.480 | 0.674 | 0.984 | – | rerank only |
| **LLM judge: Claude Sonnet 5**, rerank top-50 | **0.773** | **0.925** | 0.965 | **0.698** | **0.896** | 0.970 | – | offline / tiny rerank |
| **LLM judge: Codex gpt-6-astra**, rerank top-50 | 0.757 | 0.918 | 0.931 | 0.686 | 0.872 | 0.951 | – | offline / tiny rerank |

Free vector per skill, no text (the hyperbolic article's setup; Task B only):

| model | B: MAP | B: dir | fit MAP |
|---|---|---|---|
| box | 0.188 | 0.949 | – |
| order | 0.177 | 0.960 | 1.000 |
| order, trained with the hyperbolic (N&K) loss | 0.105 | 0.923 | 0.939 |
| TransE (h + r) | 0.285 | 0.923 | 0.884 |
| hyperbolic (article) | 0.317 | 0.500 | 0.926 |
| hyperbolic + is-a | 0.341 | 0.945 | 0.930 |
| flat twin of hyperbolic (256 dims) | 0.143 | 0.500 | 0.963 |
| flat twin + is-a | 0.268 | 0.949 | 0.955 |
| *no learning:* popularity, log(1 + #descendants) | 0.269 | 0.951 | – |
| *no learning:* neighbour vote | 0.325 | 0.762 | – |
| ***no learning:* vote + popularity** | **0.413** | 0.959 | – |

Neighbour vote: u's training neighbours (ancestors ∪ descendants), weighted by Jaccard overlap with u's own
neighbours, vote for their ancestors. Popularity breaks ties and covers pairs with no votes (weight 0.1, picked on val).

`fit MAP` = directed MAP on training pairs, i.e. relationships the model has seen (the article's headline MAP
is of this kind: 0.905–0.93 at 5–20 dims).

## Models and chosen configs

- **hyperbolic**: Poincaré ball as in the [Qdrant hyperbolic article](https://qdrant.tech/articles/hyperbolic-embeddings-qdrant/) (Mahmood & Kupchanko, 2026-09-08). The article gives
  no training details, so Nickel & Kiela 2017 loss (softmax of −distance, true ancestor vs 10 unrelated skills).
  Adam on tangent vectors via exp map at the origin instead of Riemannian SGD. Text → ball: 10 dims, 150 epochs.
  Free points: 32 dims, 300 epochs. **is-a** = N&K score −(1 + α(‖b‖ − ‖a‖))·d(a, b), α = 0.1 picked on val.
  **depth** (down direction only) = + μ·d(0, a), the profile's distance from the centre (`PoincareEmbedding.mu`; not the
  hybrid's cosine μ). Constant for a fixed query skill, so it changes no up ranking; μ picked per seed on Task D val.
- **order** (Vendrov+ 2016): f ≥ 0, a ⊑ b iff f(a) ≥ f(b) per coordinate; score −‖max(0, f(b) − f(a))‖²,
  max-margin loss. 128 dims, margin 0.1, 150 epochs.
- **+ generality + cosine** (any model M): M(a, b) + λ·log(1 + #train descendants of b) + μ·cos(a, b), (λ, μ) picked
  on val per seed from {0, 0.1, 0.3, 1} × {0, 1, 3, 10}.
- **TransE** (Bordes+ 2013): energy ‖f(a) + r − f(b)‖, one learned relation vector r, order's max-margin loss and
  negatives. Text head: 128 dims, margin 1, L1, 150 epochs. Free vectors: 32 dims, margin 4, L1.
- **flat twin of hyperbolic**: the hyperbolic model with the exp map removed and Euclidean distance. Same loss,
  negatives, optimizer and grid, plus 128 and 256 dims.
- **distilled dual**: two-role dual encoder (src/tgt heads on bge-base, score = dot product) trained with listwise
  KL to softmax(teacher/τ) over all training skills. Queries are training skills only, so no held-out skill or edge
  reaches it. Plain: 128 dims, τ 1, 150 epochs. **Folded**: the student is trained *inside* the hybrid,
  src·tgt + λ·gen + μ·cos with the teacher's (λ, μ), which is still one dot product:
  `[src(a); √μ·x(a); 1] · [tgt(b); √μ·x(b); λ·gen(b)]` (x = unit bge-base vector). 64 dims, τ 1, 300 epochs.
- **pairwise MLP**: MLP over [a, b, a⊙b, a−b] of frozen bge-base embeddings, 150 epochs.
- **cross-encoder**: BAAI/bge-reranker-base fine-tuned on (A text, B text), lr 2e-5, max 96 tokens, fp16
  inference, 3 epochs (best on val, still rising), **1 seed**. "rerank top-50" = same top-50 as the LLM judges.
- **LLM judges**: rerank the cosine + generality top-50 per test query (holds 99% / 93% of the answers) with a
  graded p(A→B) = "probability that someone with solid experience in A has basic competence in B". 51 calls
  per judge (5 queries × 50 candidates each), about 45 s (Claude) and 75 s (Codex) per call.
  Unparseable: 2 of 11,392 (Claude), 0 (Codex).
  The judged 50 are ordered by cos + gen + p and stay above the rest (`Judged`: score + 10 + p); the cross-encoder
  uses the same blend with its sigmoid output. Ordering by p alone, with cos + gen only breaking ties, scored lower on
  3 of the 4 test MAPs: Claude 0.779 / 0.681, Codex 0.726 / 0.630, against 0.773 / 0.698 and 0.757 / 0.686 for the blend.

## Findings

1. **LLM judges are far ahead**: +0.15 MAP on both tasks over the best trained model.
   - The labels were written by Claude, so a Claude judge is partly circular. But Codex, a different model
     family, scores almost the same, and the two agree closely: Spearman 0.87 over 10,648 ranking judgments,
     0.90 over 742 pair judgments.
   - Caveat: these are public tech skills the LLMs already know. The gap should shrink on private,
     company-specific skills.
   - Neither judge can be served per query (about 1 minute per 5 queries).
2. **The labels have errors the judges catch.** Both judges give p < 0.5 to 32 of the 371 true test pairs (9%):
   - 13 run through the one bad edge `JavaScript → JScript` (Vue.js, Express.js, … → JScript);
   - most of the other 19 are "built with B" mislabelled as "requires B" (`pyspark → Java`, `Logstash → Java`,
     `Elasticsearch → JVM language`, `HDF5 → C`, `Scribunto → MySQL`).
   So the LLM scores are, if anything, underestimates.
3. **Best trained model = order + generality + cosine** on A (0.618, sd 0.01) and the down search. It beats
   box + generality + cosine on A (0.582) and the cross-encoder reranker on both (0.601 / 0.480). On B (0.529, sd 0.02)
   it is behind box + gen + cos: 0.540 in the main table, 0.555 in `recall.py`. Both use the same config, grid and val
   objective, but `spike.py` trains the dual before box in each seed, so box gets different random draws. That gap is
   larger than the seed sd in `spike.py` (0.004), so treat B differences under about 0.02 as noise.
4. **Order ≈ box alone** (0.472 / 0.464 vs 0.478 / 0.501). In the hybrid, order is ahead on A. Box's remaining
   selling point is calibrated probabilities, which are **not tested** yet.
5. **TransE is weak alone, fine in the hybrid.** Alone it is the worst text model on A (0.214). One translation r
   has to map a skill to its parent *and* to its grandparent (the training pairs are the closure), so it can't
   separate levels. That is an explanation, not tested. In the hybrid it reaches 0.576 / 0.524 (≈ box + gen + cos).
   Its serving is the simplest of all: a query vector f(a) + r with Manhattan distance in plain HNSW.
   Relation conditioning (one r per relation type) **can't be tested**: this graph has one relation.
6. **Distillation into a plain-HNSW student works for B, not fully for A.**
   - The student keeps 96% of the teacher on B (0.532 vs 0.555, same seeds). A dual trained on the labels gets 0.355.
   - On A it keeps 80% (0.465 vs 0.582); folding gen + cos in lifts it to 0.496, still below cos + gen (0.520).
     Why A lags is **not isolated**.
   - The folded student is a single dot product, i.e. the only learned model here that needs no rerank. That holds
     for the up direction only: at query time in the down direction it falls to D MAP 0.089 (see "Down direction").
7. **Hyperbolic reproduces the article's fit but not generalization.** With text and the hybrid it is slightly ahead of order on B
   (0.544 vs 0.529) but trails it on A (0.547 vs 0.618) and stays symmetric: val drops the generality term (λ = 0; untested explanation:
   the ball already puts general skills near the centre). The is-a term is still needed for direction. In the down direction a depth term lifts it from last to mid-table
   (D MAP 0.331 with gen + cos, see "Down direction"). See the next
   section for why free hyperbolic vectors beat the other free-vector models, and why that doesn't matter much.
8. **Text is what generalizes**, and it is not because the child names the parent. See the section after next.

## Why hyperbolic beats the other free-vector models (tested)

| candidate cause | test | Task B MAP | verdict |
|---|---|---|---|
| curvature | flat twin: same model, loss, training; up to 256 dims | 0.143 vs 0.317 | **yes**, flat loses even with 8× the dims |
| the N&K softmax loss | order geometry trained with it | 0.105 vs 0.177 (order's own loss) | no |
| it learns graph neighbourhood + popularity | per-pair strata below; no-learning baselines | vote + popularity 0.413 | **yes**, and the explicit version is better |

hit@10 per held-out pair (Task B test, 182 pairs, ties broken at random). "Evidence" = some training neighbour of
the query skill already has the answer as an ancestor.

| model | all (182) | with evidence (96) | no evidence (86) |
|---|---|---|---|
| *no learning:* neighbour vote | 0.485 | 0.914 | 0.006 |
| *no learning:* popularity | 0.407 | 0.521 | 0.279 |
| *no learning:* vote + popularity | 0.568 | 0.878 | 0.221 |
| hyperbolic, free | 0.544 | 0.785 | 0.275 |
| TransE, free | 0.447 | 0.740 | 0.120 |
| box, free | 0.416 | 0.562 | 0.252 |
| order, free | 0.374 | 0.497 | 0.236 |
| flat twin, free | 0.233 | 0.326 | 0.128 |

- Where a neighbour already has the answer, hyperbolic finds it almost as often as the vote (0.79 vs 0.91).
- Where none does, it is no better than popularity (0.27 vs 0.28). So is every free-vector model.
- Free vectors learn neighbourhood smoothing plus a popularity axis, nothing more. Hyperbolic space does that
  smoothing best because a tree fits in it with low distortion, and flat space can't match it even at 256 dims.
- The is-a norm term is that popularity axis: it lifts the flat twin from 0.14 to 0.27, which is popularity alone.
- Caveat: MPS training is exact within a process but not across processes. Rerunning the probe moved free
  hyperbolic by up to 0.04 (0.83 → 0.79 with evidence) and order-with-N&K-loss more; the other models didn't move.
- Practical upshot: without text, don't train anything. Vote + popularity (a sparse matrix product) beats every
  free-vector model.

## Why text generalizes (tested, correlational)

Same pairs, split by whether the parent's label appears in the child's encoder text ("named", 75 pairs).

| model | named (75) | not named (107) | not named, no graph evidence (48) |
|---|---|---|---|
| cosine | 0.480 | 0.150 | 0.146 |
| cosine + generality | 0.827 | 0.664 | 0.562 |
| box, text | 0.720 | 0.729 | 0.653 |
| order, text | 0.733 | 0.713 | 0.611 |
| hyperbolic, text | 0.631 | 0.651 | 0.465 |
| hyperbolic, free | 0.529 | 0.555 | 0.292 |
| vote + popularity | 0.560 | 0.573 | 0.229 |

- Raw cosine depends on the name (0.48 vs 0.15). Trained text models don't (box 0.72 vs 0.73), so they are not
  string matching.
- The text advantage sits where the graph has nothing: box 0.65 vs 0.29 (free hyperbolic) and 0.23 (vote + popularity)
  on the 48 pairs with neither a name nor a neighbour (±0.07 at this n).
- Most of it is already in the pretrained encoder plus popularity (cos + gen 0.56). Training adds about 0.09.
- Correlational only. The causal test is to strip parent names and synonyms from child texts and retrain.

## Beyond MAP: recall, generalization, cost (`experiments/recall.py`)

Same test sets, 3 seeds. R@K = share of true answers in the top K of the 925 candidate skills
(K = 50 / 100 is 5% / 11% of the catalogue). "No evidence" = hit@10 on the 86 Task B pairs that no graph neighbour of the
query already has, i.e. not derivable from the graph. Train s = wall time per seed on an M5 Pro (MPS).
Small differences from the main table (box up to 0.025) are different random draws (`spike.py` trains the dual before box in each seed) and MPS noise, the tie-break fix below (order, box, vote, popularity: ≤ 0.02) and the val filter (see "Val / test overlap"; order + gen + cos 0.622 / 0.546 → 0.618 / 0.529).

| method | A R@10 | A R@50 | A R@100 | B R@10 | B R@50 | B R@100 | no evidence | train s |
|---|---|---|---|---|---|---|---|---|
| cosine | 0.459 | 0.690 | 0.798 | 0.307 | 0.648 | 0.748 | 0.360 | 0 |
| popularity | 0.377 | 0.745 | 0.899 | 0.386 | 0.710 | 0.868 | 0.279 | 0 |
| cosine + generality | 0.770 | **0.991** | 0.991 | 0.760 | 0.929 | 0.963 | 0.616 | 0 |
| neighbour vote | – | – | – | 0.498 | 0.585 | 0.594 | 0.006 | 0 |
| vote + popularity | – | – | – | 0.567 | 0.801 | 0.898 | 0.221 | 0 |
| hyperbolic, free vectors | – | – | – | 0.538 | 0.814 | 0.925 | 0.260 | 8 |
| dual | 0.679 | 0.914 | 0.963 | 0.645 | 0.864 | 0.911 | 0.481 | 3 |
| box | 0.773 | 0.939 | 0.980 | 0.736 | 0.922 | 0.954 | 0.601 | 4 |
| box + gen + cos | 0.810 | 0.970 | 0.982 | 0.776 | 0.941 | 0.966 | **0.674** | 4 |
| order | 0.794 | 0.965 | 0.982 | 0.730 | 0.918 | 0.961 | 0.585 | 3 |
| order + gen + cos | **0.878** | 0.989 | **0.999** | **0.782** | 0.955 | 0.975 | 0.640 | 3 |
| hyperbolic (text) | 0.605 | 0.895 | 0.963 | 0.661 | 0.905 | 0.950 | 0.422 | 5 |
| hyperbolic + gen + cos | 0.779 | 0.982 | 0.993 | 0.768 | **0.969** | **0.985** | 0.601 | 5 |
| TransE | 0.424 | 0.763 | 0.846 | 0.597 | 0.838 | 0.889 | 0.333 | 3 |
| TransE + gen + cos | 0.803 | 0.975 | 0.984 | 0.767 | 0.962 | 0.980 | 0.659 | 3 |
| pairwise MLP | 0.806 | 0.979 | 0.992 | 0.634 | 0.908 | 0.942 | 0.484 | 4 |
| distilled dual, folded | 0.720 | 0.929 | 0.956 | 0.765 | 0.927 | 0.953 | 0.659 | 15 (teacher + student) |

- **Recall at prefetch depth is nearly solved.** Cosine + generality with no training holds 99% (A) / 93% (B) of
  answers in its top 50. The hybrids reach 94–97% on B. So the first stage is cheap, and precision (MAP, R@10) is where
  the methods differ.
- **Hyperbolic + gen + cos has the best deep recall on B** (0.969 at 50, 0.985 at 100) despite a middling MAP. It is a
  good prefetch, not a good final ranker.
- **Graph-only methods hit a ceiling.** The vote never passes 0.67 even at K = 200, because a third of the answers get
  no vote at all. Popularity fills the gap to 0.91.
- **Generalization to pairs the graph can't derive:** box + gen + cos 0.674, then the folded distilled dual and TransE +
  gen + cos (0.659), and order + gen + cos (0.640). Free vectors and the vote are at 0.01–0.28.
- **Training cost is negligible** here (3–15 s per model). In a new domain the real costs are labels and choosing an encoder.

## Down direction: requirement → who has it (`experiments/down.py`)

Tasks A and B only measure the up direction (skill → what it implies). A search like "who knows FP?" runs down.

- **Task D.** Each of the 69 training skills that some unseen test skill implies ranks 1035 candidates (925 training + 110 test skills).
  - Relevant: the test skills that imply it.
  - The requirement's own profile and the training skills that imply it are filtered out, so the task is finding unseen profiles among the rest of the catalogue.
  - The first version left the requirement's own profile in as a negative (fixed 2026-09-28 after the Codex review). Cosine scores that profile 1, so it always took rank 1 and pushed every relevant profile down one place. The fix lifted cosine from 0.170 to 0.274 and order + gen + cos from 0.229 to 0.366 (direct) and from 0.378 to 0.498 (via expansion). Old numbers: `results/pre_selffix/`.
- **Two routes are compared:**
  - *Query-time*: score(profile, requirement) directly.
  - *Expanded*: each profile ranks the catalogue in the tested up direction, and the requirement then ranks profiles by where it falls in their lists. This is what storing each profile's top-K implied skills at ingest gives you.
- Configs, seeds and (λ, μ) are the same as in `recall.py`. In the down direction λ·gen(requirement) is a constant, so cos + gen ranks exactly like cosine at query time. A query-time route with (λ, μ) tuned for the down search is untested. The one down-tuned weight is hyperbolic's depth μ in the "+ depth" rows (picked per seed on Task D val from 0–2: 1.1 / 1.5 / 1.1 alone, 1.5 / 1.3 / 1.1 inside gen + cos).
- **Ties are broken at random** (`rank_metrics`, since 2026-09-28). Before, `argsort` broke them by candidate position, and in Task D every relevant candidate sits at the end of the list.
  - The expanded scores tie often, because many profiles place a requirement at the same rank.
  - Breaking ties by the raw score, as the first version of this table did, scored below random (0.273 vs 0.313 before the self-profile fix): among profiles that share a rank, the direct down score is worse than chance.
  - Up-direction numbers moved ≤ 0.02 (`recall.py` rerun). The main table and the val sweeps predate the fix.

| model | down, direct: MAP | R@10 | R@50 | down, via expansion: MAP | R@10 | R@50 |
|---|---|---|---|---|---|---|
| cosine | 0.274 | 0.412 | 0.689 | 0.396 | 0.537 | 0.745 |
| cos + gen | 0.274 | 0.412 | 0.689 | 0.436 | 0.622 | 0.815 |
| dual | 0.091 | 0.186 | 0.507 | 0.103 | 0.232 | 0.579 |
| distilled dual, folded | 0.089 | 0.126 | 0.241 | 0.257 | 0.453 | 0.709 |
| box | 0.219 | 0.377 | 0.641 | 0.249 | 0.442 | 0.755 |
| box + gen + cos | 0.350 | 0.530 | 0.726 | 0.449 | 0.640 | 0.795 |
| order | 0.181 | 0.325 | 0.657 | 0.314 | 0.533 | 0.759 |
| order + gen + cos | **0.366** | **0.552** | 0.769 | **0.498** | **0.642** | 0.832 |
| hyperbolic | 0.019 | 0.015 | 0.142 | 0.255 | 0.371 | 0.674 |
| hyperbolic + is-a | 0.019 | 0.015 | 0.142 | 0.250 | 0.373 | 0.674 |
| hyperbolic + gen + cos | 0.056 | 0.097 | 0.327 | 0.423 | 0.599 | **0.866** |
| hyperbolic + depth | 0.195 | 0.325 | 0.635 | 0.255 | 0.371 | 0.674 |
| hyperbolic + gen + cos + depth | 0.331 | 0.516 | **0.794** | 0.423 | 0.599 | **0.866** |
| TransE | 0.083 | 0.152 | 0.292 | 0.102 | 0.198 | 0.385 |
| TransE + gen + cos | 0.312 | 0.460 | 0.649 | 0.447 | 0.620 | 0.789 |
| pairwise MLP | 0.270 | 0.463 | 0.685 | 0.412 | 0.624 | 0.830 |

- **Down is much harder than up.** The best query-time model, order + gen + cos, reaches D MAP 0.366, against 0.274 for cosine; its up-direction A MAP is 0.618. Only the + gen + cos hybrids of box, order and TransE, and hyperbolic + gen + cos + depth, beat cosine at query time.
- **Expansion wins for every model.** Order + gen + cos goes from 0.366 to 0.498. Cos + gen expanded (no training) reaches 0.436, above every query-time model.
- **TransE and both duals collapse at query time.** Folded dual: 0.089 (0.257 expanded). TransE 0.083 and the dual 0.091 stay near 0.1 expanded too. A guess for the folded dual, untested: the student was distilled with a per-query softmax over candidates, and adding a constant to one profile's scores leaves that loss unchanged, so nothing constrains how profiles compare for a fixed requirement.
- **Hyperbolic collapsed too (0.019; 0.056 with gen + cos) mostly because its distance is symmetric; a depth term on the profile fixes most of that** (0.195; 0.331 with gen + cos).
  - Cause: in the symmetric top 10, 88% of the wrong profiles are *more general* than the requirement (3 seeds; a one-off check, not in `experiments/`). The right ones are more specific, and symmetric distance can't tell the two apart.
  - μ·d(0, profile) rewards being deeper in the ball. At μ = 1 the score is d(0, a) − d(a, b), which peaks when the requirement b lies on the path from the centre to the profile a, as an ancestor does in a tree.
  - Hyperbolic + gen + cos + depth has the best direct R@50 (0.794), but its MAP (0.331) is below order + gen + cos (0.366) and box + gen + cos (0.350), and its expanded route is unchanged (0.423).
  - The same term doesn't help the up direction: there the right answers and 100% of the wrong ones are *more general* than the query, so depth can't separate them (val picks a weight of 0 for the up analogue, −λ·d(0, b)).
  - Not yet served: it needs a Formula Query rerank with d(0, a) from the stored `sq_norm`, and a prefetch wide enough to hold the deep profiles.
- → **Serve by expanding at ingest**: store implied skills as a sparse vector or payload per profile. A query-time down search over these heads isn't competitive. Expanding at ingest also makes the LLM affordable, because it runs once per profile, not once per query.

**CV demo (`showcase.ipynb`, 6 hand-written CVs, my labels: an anecdote, not a measurement).**
- The query "Programming in the functional programming paradigm" links by cosine to the "functional programming language" node (0.81).
- The Scala CV's expansion under the hybrids contains Scala, JVM language, Java and FP, but also Python and ruby (from the generality prior).
- FP also ranks high for the non-FP CVs (Python data scientist, Java/Spring).
- Every embedding model is fooled by a recruiter CV that only names FP languages: FP ranks 1–2 in its expansion, and it lands in the top 2 at query time.
- The LLM separates the two groups: 0.85–0.99 for the FP CVs, 0.15–0.2 for the rest, including the trap.
- The heads were trained on encyclopedia descriptions of skills, not CVs, so CV text is off-distribution for them.

## Served from Qdrant (`experiments/qdrant_serve.py`)

The expanded route, run end to end in Qdrant 1.18 (local podman container, qdrant-client 1.19).
- **Ingest.**
  - Qdrant inference (`models.Document`, FastEmbed in the client) embeds each text; its vectors match the cached sentence-transformers ones (cos 1.000 on 200 nodes). On Qdrant Cloud the same call runs server side; that path hasn't run.
  - A head (order + gen + cos seed 0, or cos + gen) expands each profile into its top K implied catalogue skills, stored as a sparse vector `<head>@<K>`: index = skill id, value = 926 − rank (rank 1 = top).
  - Collections: `skills` (925 catalogue skills, dense) and `profiles` (1035 Task D profiles: dense text plus one sparse vector per head and K). Ingest took 82 s, nearly all of it CPU embedding.
- **Query.** Requirement text → dense search in `skills` → skill id → sparse query `{skill: 1}` on `profiles`. Several requirements = several indices, and their values add. Payload filters (e.g. `kind = cv`) work as usual.

| route (seed 0) | D MAP | R@10 | R@50 | R@100 |
|---|---|---|---|---|
| dense, direct | 0.274 | 0.412 | 0.689 | 0.775 |
| order+gen+cos, torch | 0.491 | 0.622 | 0.816 | 0.904 |
| order+gen+cos@10 | 0.451 | 0.564 | 0.679 | 0.748 |
| order+gen+cos@50 | 0.490 | 0.622 | 0.787 | 0.875 |
| order+gen+cos@100 | 0.491 | 0.622 | 0.801 | 0.889 |
| order+gen+cos@925 | 0.491 | 0.622 | 0.816 | 0.904 |
| cos+gen, torch | 0.436 | 0.622 | 0.815 | 0.863 |
| cos+gen@10 | 0.379 | 0.506 | 0.609 | 0.649 |
| cos+gen@50 | 0.436 | 0.622 | 0.815 | 0.863 |
| cos+gen@100 | 0.436 | 0.622 | 0.815 | 0.863 |
| cos+gen@925 | 0.436 | 0.622 | 0.815 | 0.863 |
| order+gen+cos@100, label text linked | 0.443 | 0.561 | 0.739 | 0.818 |

- **@925 gives the torch metrics** for both heads: Qdrant's scores differ from torch's by a constant, and `rank_metrics` breaks ties the same random way.
- **K = 100 per profile loses no MAP**, but R@50 / R@100 drop 1.5 points for order + gen + cos (0.816 → 0.801, 0.904 → 0.889), K = 50 loses ≤ 0.001 MAP, and K = 10 costs 0.04–0.06. Profiles past K aren't returned; they tie last and `rank_metrics` orders them at random, so @K metrics give a little credit to profiles Qdrant never returned.
- **Starting from free label text costs 0.05** (0.443). Dense search links a label to its own skill 85.5% of the time; the misses are the loss.
- **p50 latency is 10.5 ms** for embed + link + sparse top 10, measured on the 1035 Task D profiles (load average 6.7 during the run; the same query path measured 7.7 ms earlier the same day, and a repeat gave 10.8 ms). Scale is untested.
- In the notebook's CV demo (section 10), Qdrant shows the same weakness as in torch: FP is in the top 10 of every CV's expansion, so K = 100 returns all six, and the recruiter trap ties the Haskell CV at rank 1.

## Identity check (`experiments/identity.py`)

Every skill implies itself, so a search for skill b should return b's own profile first. Per catalogue skill b (925), same models, seeds and (λ, μ) as `down.py`; ties count half, so "top 1" means alone at the top.
- **up top 1**: b ranks first in its own implied list. **in top 100**: b is inside the K = 100 list Qdrant stores; past it, b's own profile never comes back for b.
- **down direct / expanded top 1**: b's own profile ranks first of the 1035 Task D profiles for requirement b.

| model | up top 1 | up median rank | in top 100 | down direct top 1 | down expanded top 1 | expanded top 10 |
|---|---|---|---|---|---|---|
| cosine | 1.000 | 1 | 1.000 | 1.000 | 0.896 | 1.000 |
| cos+gen | 0.995 | 1 | 1.000 | 1.000 | 0.960 | 1.000 |
| dual | 0.008 | 141 | 0.406 | 0.015 | 0.026 | 0.207 |
| box | 0.077 | 9.3 | 1.000 | 0.584 | 0.761 | 0.983 |
| box+gen+cos | 0.994 | 1 | 1.000 | 0.996 | 0.940 | 1.000 |
| order | 0.343 | 1.5 | 1.000 | 0.841 | 0.924 | 0.992 |
| order+gen+cos | 1.000 | 1 | 1.000 | 1.000 | 0.951 | 1.000 |
| hyp | 1.000 | 1 | 1.000 | 1.000 | 0.981 | 0.999 |
| hyp+gen+cos | 1.000 | 1 | 1.000 | 1.000 | 0.970 | 1.000 |
| transe | 0.002 | 192 | 0.231 | 0.003 | 0.012 | 0.177 |
| transe+gen+cos | 0.765 | 1.3 | 1.000 | 0.959 | 0.898 | 0.996 |
| pair_mlp | 0.405 | 2.7 | 0.976 | 0.979 | 0.953 | 0.993 |

- **Both picks pass.** order + gen + cos and box + gen + cos keep b first in its own list and first in the direct down score, and b's own profile is in the expanded top 10 for every requirement.
- **Expanded top 1 below 1 is mostly a tie, not a miss**, for heads with up top 1 ≥ 0.99: nearly every catalogue profile ranks itself first, so the profile sharing rank 1 with b's own is nearly always an unseen test profile whose top skill is b. Whether those tie partners are true matches is untested.
- **order alone ties b with its ancestors** (order score is 0 for self and for every ancestor it contains; median rank 1.5). The +gen+cos terms break the tie.
- **box alone ranks general ancestors above b itself** (median rank 9). The cause is untested; a guess is the soft box intersection, which makes a box's overlap with itself smaller than a big box's overlap with it.
- **hyp + depth and hyp + gen + cos + depth give the same row as hyp and hyp + gen + cos**: with μ up to 1.5, b's own profile still comes first in the direct down score.
- **TransE and the dual fail by construction**: TransE's self score is −‖r‖, and a dot product between separate source and target vectors has no reason to peak on the diagonal (the dual part is a guess). Their own profile falls outside the stored top 100 for 77% and 59% of skills.

## What is tested and what isn't

**Tested** (numbers above):
- every row in both tables: 3 seeds except the cross-encoder, configs picked on val, direction for every model;
- hyperbolic reproduces the article's fit (MAP 0.934 on seen pairs, 32 dims);
- the is-a norm term fixes hyperbolic direction (0.50 → 0.97) at no ranking cost;
- a depth term on the profile fixes hyperbolic's down-direction collapse (D MAP 0.019 → 0.195; 0.056 → 0.331 with gen + cos);
- why hyperbolic wins among free vectors (curvature, not the loss; it approximates vote + popularity);
- text models don't depend on the child naming the parent; their edge is on pairs with no graph evidence;
- TransE alone and in the hybrid; distillation into a dual, plain and folded;
- recall at prefetch depth (R@50/100/200) and training time for every indexable method (`experiments/recall.py`);
- the down direction (requirement → unseen profiles) at query time and through ingest-time expansion (`experiments/down.py`);
- the expanded route served from Qdrant: same metrics as torch at K = 925, the effect of K, free-text linking, p50 latency on the 1035-profile collection (`experiments/qdrant_serve.py`);
- the cos + gen top-50 holds 99% (A) / 93% (B) of test answers, which is the rerank ceiling;
- the two LLM judges agree (Spearman 0.87 / 0.90); cost and latency per call;
- 32 true test pairs are rejected by both judges; 13 of them go through `JavaScript → JScript`.

**Partly tested**:

| claim | what we have | what would settle it |
|---|---|---|
| the 32 rejected pairs are label errors | two LLM families agree | your pass over `docs/audit_100.csv` |
| the Claude judge isn't just echoing Claude's labels | Codex scores almost the same | human labels on a sample |
| text generalizes through world knowledge | strata above (correlational) | strip parent names and synonyms from child texts, retrain |
| cross-encoder numbers | 1 seed, 3 epochs, still improving | 3 seeds, more epochs |
| seed sds cover all run-to-run noise | MPS is not bit-reproducible across processes; free hyperbolic moves ≈ ±0.03 | rerun on CPU, or report sd over processes |
| why TransE fails on A | closure argument | per-depth hit rates |
| why the distilled student lags on A | fold helps only +0.03 | distill from order + gen + cos; add held-out-style queries |

**Not tested**:

| claim | what it needs |
|---|---|
| the rest of the "Qdrant serving" column (the query-time routes): ANN recall vs exact, range filters for box/order containment, Formula Query for generality and hyperbolic, the folded dot product; any route at 10k+ profiles; Qdrant Cloud inference | load those vectors, compare ANN top-k to exact top-k, measure p50/p99 at scale |
| box probabilities are calibrated P(B \| A) | graded labels (LLM relabel) and a reliability curve |
| profile ⇒ project coverage with several skills (box intersection, order max) | a set of profiles and projects with judged matches |
| private skills with thin text; CV-style profile text | company skills, or descriptions reduced to labels; a set of real CVs with judged skills (the notebook's 6 CVs are an anecdote) |
| LLM expansion at ingest beats the heads on CVs | the same CV set, LLM vs heads |
| why the folded dual collapses at query time in the down direction | distill with a column-wise loss too; per-depth analysis |
| a query-time down route with (λ, μ) tuned for the down search (only hyperbolic's depth μ is down-tuned) | a val split of down queries and the same HW grid |
| LLM as teacher (graded relabel, then train/distill) | your go; about 27 calls per judge |
| results hold beyond popular skills, and for held-out skills whose parent is also new | popularity-stratified and new-parent splits |
| hyperbolic's edge grows on bigger, deeper hierarchies | a second graph (e.g. ESCO) |
| relation-conditioned TransE | a graph with two or more relation types |
| recall at a fixed K holds on a bigger catalogue (here K = 50 is 5% of 925 skills) | a catalogue of 10k+ skills |
| the pipeline transfers to another domain with the same structure | a second labelled graph; only `data/graph.json` has to change |
| go / no-go, rubric (`docs/satisfies_rubric.md`) | yours |

## Suggested next step

1. Relabel all 2,630 candidate edges with graded p from both judges (about 27 calls each). This also fixes the
   JScript and "built with" errors, and gives calibration targets.
2. Train order / box on the soft labels, and distill order + gen + cos (or the LLM itself) into the folded dual.
3. The expanded route now runs in Qdrant (`experiments/qdrant_serve.py`). What's left is scale (10k+ profiles, p99) and a real CV
   set, the weak spot: every embedding head is fooled by CV text, and the LLM isn't.

## Reproduce

Code layout: `skillmatch/` is the library (data, training loops, metrics, tasks, Qdrant serving) with one file per
method in `skillmatch/methods/`; `experiments/` holds the runs behind `results/*.json` (`zoo.py` = configs picked on
val); `pipeline/` builds `data/graph.json`; `tools/` builds the notebooks. Run from the repo root:

```
uv run python -m experiments.spike            # cosine, cos+gen, dual, box, box+gen+cos  → results/spike.json
uv run python -m experiments.compare sweep    # val grid → results/compare_sweep.json
uv run python -m experiments.compare test     # 3 seeds, incl. +gen+cos rows → results/compare.json
uv run python -m experiments.compare ce       # cross-encoder → results/compare.json
uv run python -m experiments.compare distill [fold]   # distillation into the dual → results/compare.json
uv run python -m experiments.llm_judge        # both judges (cached in data/llm_cache/) → results/llm_judge.json
uv run python -m experiments.probe            # strata + no-learning baselines → stdout (results/probe.log)
uv run python -m experiments.recall           # R@K, no-evidence hit@10, train time → results/recall.json
uv run python -m experiments.down             # down direction, query-time vs expanded → results/down.json
uv run python -m experiments.qdrant_serve     # needs Qdrant on :6333; expanded route served from Qdrant → results/qdrant.json
uv run python -m experiments.identity         # does b's own profile top the search for b? → results/identity.json
uv run --with jupyter jupyter lab showcase.ipynb   # tables, live models, demos, Qdrant part (~3 min to run)
uv run python -m experiments.compare table
```
