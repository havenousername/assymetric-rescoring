"""Untrained baselines on the frozen text vectors (unit length, so the dot product is the cosine).

cosine:         s(A→B) = cos(x_A, x_B). Symmetric, so it can't tell A → B from B → A.
cos + gen:      s(A→B) = cos(x_A, x_B) + λ·generality(B), generality = log(1 + #training descendants).
                Requirements tend to be general skills, so a count of known descendants is a cheap direction signal.
Serving: one dense vector per skill (cosine distance); cos + gen adds the stored count through a score formula.
"""
LAMBDA = 0.05  # picked on val (experiments/spike.py)


def cosine(X):
    return lambda a, b: X[a] @ X[b].T


def cosine_generality(X, generality, lam=LAMBDA):
    return lambda a, b: X[a] @ X[b].T + lam * generality[b]
