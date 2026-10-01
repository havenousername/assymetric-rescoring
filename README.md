# asymmetric-skills

Skill similarity is symmetric, but skill *implication* is not. Someone who knows Spring Boot knows Java; someone who knows Java may never have touched Spring Boot. This repo asks: **can we score `s(A→B)`, "having A implies having B", and serve it from [Qdrant](https://qdrant.tech)?**

Plain cosine similarity can't do it. It scores both directions the same, so it picks the right direction half the time (`dir` = 0.5). Adding a per-skill *generality* term and an order or box embedding fixes the direction (`dir` ≈ 0.98) and roughly doubles ranking quality (test MAP 0.27 → 0.62 on the main task). Full table: [`docs/results.md`](docs/results.md).

## What's here

| Path | What it is |
|---|---|
| `pipeline/` | Builds the dataset: Wikidata + Stack Overflow tags + Wikipedia extracts → candidate pairs → labelled implication DAG → transitive closure → train/val/test splits (`data/graph.json`). |
| `skillmatch/` | The library. `methods/` has one module per scorer (cosine, dual, box, order, TransE, hyperbolic, pair-MLP, cross-encoder, distillation); `data`, `training`, `metrics`, `tasks` and `qdrant` hold splits, evals and serving. |
| `experiments/` | The scripts behind `results/*.json`, such as `compare.py`, `spike.py`, `zoo.py` and `qdrant_serve.py`. Logs sit next to the results. |
| `cvsearch/` | The same idea on real data. It mines keywords from Djinni CVs, builds an implication graph from co-occurrence, and searches so that a request for "Java" also finds CVs that only name "Spring Boot". |
| `tools/` | Generators for the notebooks and the static demo page. |
| `showcase.ipynb`, `methods.ipynb`, `cvsearch.ipynb` | Walkthroughs: the problem, every method trained and served from Qdrant, and the CV search. |

## Run it

```bash
uv sync
docker run -p 6333:6333 qdrant/qdrant      # needed for methods.ipynb and the serving evals
uv run jupyter lab                         # open one of the notebooks
uv run python -m experiments.compare test  # example: re-run the model comparison
```

The Djinni CVs (about 226 MB) are downloaded on first use by `cvsearch.data` into `data/djinni/`, which is git-ignored.

## Reading the results

- **Task A**: rank training skills as ancestors of an unseen skill. **Task B**: filtered ranking of held-out implied pairs.
- **MAP** is the main score. **dir** is the share of true pairs scored higher in the true direction, so 0.5 means a coin flip.
- Learned models are 3-seed means, with configs picked on validation only.

## License

[MIT](LICENSE). The Djinni CV dataset is MIT-licensed too (Drushchak & Romanyshyn, 2024).
