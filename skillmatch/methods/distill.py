"""Distillation into a dual encoder, so a structured teacher (box + gen + cos) serves as one dot product.
Listwise KL: for each catalogue skill, the student's softmax over all catalogue skills matches softmax(teacher / τ).
Only catalogue skills are queries and candidates, so held-out skills and hidden edges never reach the student.
fold = (λ, μ): the student is src·tgt + λ·gen + μ·cos and learns only the box part. It stays one dot product:
[src(A), √μ·x_A, 1] · [tgt(B), √μ·x_B, λ·gen(B)]."""
import torch
import torch.nn.functional as F

from ..data import DEVICE
from .dual import DualEncoder
from .hybrid import hybrid


def distill(graph, teacher, dim=64, tau=1.0, epochs=150, lr=1e-3, batch=64, fold=None):
    """Returns (student, its scorer)."""
    catalogue = torch.tensor(graph.catalogue)
    with torch.no_grad():
        T = torch.cat([teacher(catalogue[i:i + 256], catalogue) for i in range(0, len(catalogue), 256)])
        P = F.softmax(T.fill_diagonal_(-float("inf")) / tau, -1)
    student = DualEncoder(dim).to(DEVICE)
    opt = torch.optim.Adam(student.parameters(), lr=lr)
    S = student.scorer(graph.X) if fold is None else hybrid(student.scorer(graph.X), graph, *fold)
    for _ in range(epochs):
        for q in torch.randperm(len(catalogue)).split(batch):
            loss = F.kl_div(F.log_softmax(S(catalogue[q], catalogue), -1), P[q.to(DEVICE)], reduction="batchmean")
            opt.zero_grad(); loss.backward(); opt.step()
    return student.eval(), S


def folded_vectors(student, x, generality, lam, mu):
    """(query, document) vectors whose dot product is the folded score: query for the A side, document for B."""
    ones = torch.ones(len(x), 1, device=x.device)
    query = torch.cat([student.src(x), mu ** 0.5 * x, ones], -1)
    document = torch.cat([student.tgt(x), mu ** 0.5 * x, lam * generality[:, None]], -1)
    return query, document
