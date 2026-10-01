"""LLM judges rerank the cos + gen top 50 of every test query (skillmatch.methods.llm_judge).
Raw answers cached in data/llm_cache/; metrics merged into results/llm_judge.json.
Usage: uv run python -m experiments.llm_judge [judge ...]"""
import json
import sys

import torch

from skillmatch.data import SkillGraph
from skillmatch.methods.llm_judge import JUDGES, Judged, judge_all
from skillmatch.tasks import evaluate

from .zoo import load, save

if __name__ == "__main__":
    graph, res = SkillGraph(), load("llm_judge.json")
    for judge in sys.argv[1:] or JUDGES:
        rank, pair, missing = judge_all(graph, judge)
        with torch.no_grad(): res[judge] = {**evaluate(graph, Judged(graph, rank, pair), "test"), "missing": missing}
        print(judge, json.dumps(res[judge], default=float), file=sys.stderr)
        save("llm_judge.json", res)
