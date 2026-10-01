"""The heads' scores as Qdrant Formula Query expressions, for geometries Qdrant has no distance for.

Pattern: store each catalogue skill's head representation in the payload (an array field), prefetch a shortlist by
cosine on the text vector, and let the formula rescore the shortlist server side. Each function mirrors a torch
score in skillmatch/methods: a is the query skill's representation (a 1-D tensor), key the payload field holding
the stored skill's. "$score" is the prefetch's cosine and "gen" the stored generality, so any head plus the free
signals is one formula (hybrid).
Formulas have no max and no log-sum-exp, so those are rebuilt from abs, exp and ln in overflow-safe forms: Qdrant
rejects a query whose formula yields inf or NaN.
"""
from qdrant_client import models

from .methods.box import EULER_GAMMA


def total(terms):
    return models.SumExpression(sum=list(terms))


def times(*factors):
    return models.MultExpression(mult=list(factors))


def absolute(x):
    return models.AbsExpression(abs=x)


def square(x):
    return models.PowExpression(pow=models.PowParams(base=x, exponent=2))


def ln(x):
    return models.LnExpression(ln=x)


def log1p_exp_neg_abs(x):
    """ln(1 + e^−|x|), in [0, ln 2]: the correction term of softplus and log-sum-exp."""
    return ln(total([1, models.ExpExpression(exp=times(-1, absolute(x)))]))


def relu(x):
    """max(0, x) = (x + |x|) / 2."""
    return times(0.5, total([x, absolute(x)]))


def maximum(x, c):
    """max(x, c) for a constant c."""
    return times(0.5, total([x, c, absolute(total([x, -c]))]))


def minimum(x, c):
    """min(x, c) for a constant c."""
    return times(0.5, total([x, c, times(-1, absolute(total([x, -c])))]))


def logaddexp(c, x):
    """log(e^c + e^x) for a constant c: max(c, x) + ln(1 + e^−|x − c|)."""
    return total([maximum(x, c), log1p_exp_neg_abs(total([x, -c]))])


def softplus(x):
    """log(1 + e^x) = max(0, x) + ln(1 + e^−|x|)."""
    return total([relu(x), log1p_exp_neg_abs(x)])


def field(key, i):
    return f"{key}[{i}]"


def cos_gen(lam):
    """s(A→B) = cos(A, B) + λ·gen(B): the prefetch score plus the stored generality."""
    return total(["$score", times(lam, "gen")])


def hybrid(head, lam, mu):
    """s(A→B) = head + λ·gen(B) + μ·cos(A, B), as methods.hybrid."""
    return total([head, times(lam, "gen"), times(mu, "$score")])


def order(a, key="order"):
    """OrderEmbedding.score: −Σᵢ max(0, bᵢ − aᵢ)²."""
    return times(-1, total(square(relu(total([field(key, i), -v]))) for i, v in enumerate(a.tolist())))


def box(model, a, key="box"):
    """BoxEmbedding.score: log vol(A ∩ B) − log vol(A), at most −1e-6. payload[key] = B's [lo | hi]."""
    d, bi, bv = len(a) // 2, model.beta_i, model.beta_v
    lo_a, hi_a = a[:d].tolist(), a[d:].tolist()
    log_vol_a = model.log_volume(a[:d], a[d:]).item()
    log_sides = []
    for i in range(d):
        lo = times(bi, logaddexp(lo_a[i] / bi, times(1 / bi, field(key, i))))  # soft max of the lower corners
        hi = times(-bi, logaddexp(-hi_a[i] / bi, times(-1 / bi, field(key, d + i))))  # soft min of the upper corners
        side = total([hi, times(-1, lo), -2 * EULER_GAMMA * bv])
        log_sides.append(ln(total([times(bv, softplus(times(1 / bv, side))), 1e-10])))
    return minimum(total([*log_sides, -log_vol_a]), -1e-6)


def poincare(model, a, key="hyp"):
    """PoincareEmbedding.score: −(1 + α(‖B‖ − ‖A‖))·d(A, B), d = acosh(1 + 2‖A − B‖² / ((1 − ‖A‖²)(1 − ‖B‖²))).
    payload: key = B's coordinates, key + "_sq" = min(‖B‖², 1 − 1e-5), key + "_norm" = ‖B‖.
    The depth term μ·d(0, A) is left out: A is the query, so it is a constant here."""
    uu = min((a @ a).item(), 1 - 1e-5)
    dist2 = total(square(total([field(key, i), -v])) for i, v in enumerate(a.tolist()))
    denominator = times(1 - uu, total([1, times(-1, f"{key}_sq")]))
    x = maximum(total([1, models.DivExpression(div=models.DivParams(left=times(2, dist2), right=denominator))]), 1 + 1e-7)
    d = ln(total([x, models.SqrtExpression(sqrt=total([square(x), -1]))]))  # acosh
    return times(-1, total([1, times(model.alpha, total([f"{key}_norm", -a.norm().item()]))]), d)


def transe(a, model, key="transe"):
    """TransE.score: −‖f(A) + r − f(B)‖_p, a = f(A). payload[key] = f(B)."""
    diffs = [total([field(key, i), -v]) for i, v in enumerate((a + model.r).tolist())]
    return times(-1, total(map(absolute, diffs)) if model.p == 1 else models.SqrtExpression(sqrt=total(map(square, diffs))))
