"""Any scorer plus the two free signals: s(A→B) = M(A, B) + λ·generality(B) + μ·cos(A, B).
The heads learn from under a thousand edges and forget some of what the encoder knows; cosine brings text
similarity back and generality adds the prior that requirements are general skills. (λ, μ) are picked on val."""
from ..tasks import val_score

GRID = [(lam, mu) for lam in (0, 0.1, 0.3, 1) for mu in (0, 1, 3, 10)]


def hybrid(M, graph, lam, mu):
    return lambda a, b: M(a, b) + lam * graph.generality[b] + mu * (graph.X[a] @ graph.X[b].T)


def tune(graph, M, grid=GRID):
    """The (λ, μ) with the best val score."""
    return max(grid, key=lambda w: val_score(graph, hybrid(M, graph, *w)))
