"""CV search with implied skills, built from the CVs alone: keywords mined from the CVs, an implication graph from
their co-occurrence, five scorers trained on it, and a search that lists CVs naming the requirement first.

    graph, _ = build_graph(graph_cvs())             # vocabulary + implication graph, no labels
    heads = {h: train(graph, h, epochs) for h in HEADS}
    search = Search(CVIndex(eval_cvs("test"), graph.vocab), scorers(graph, heads, weights))
    search.query("Looking for the C++ experts", mode="down")
"""
from .data import TECH, Vocabulary, eval_cvs, graph_cvs
from .graph import Cooccurrence, KeywordGraph, build_graph, nli_judge, subsumption_edges, verify
from .models import HEADS, NAMES, embed_keywords, graph_lookup, scorers, train
from .search import UP_CHECK, CVIndex, Search, column, cv_scores, grid, targets, tune, up_check
