"""Identity check (skillmatch.tasks.identity): every skill implies itself, so a search for b should return b's own
profile first. Ties count half. Same models, configs, seeds and (λ, μ) as down.py.
Usage: uv run python -m experiments.identity → results/identity.json"""
from skillmatch.data import SkillGraph
from skillmatch.tasks import identity

from .down import collect, runs

if __name__ == "__main__":
    graph = SkillGraph()
    res = collect(runs(graph, solo=("pair_mlp",), distilled=False), lambda name, M: {name: identity(graph, M)}, "identity.json")
    keys = list(next(iter(res.values())))
    print(f"{'model':18}" + "".join(f"{k:>22}" for k in keys))
    for name, r in res.items(): print(f"{name:18}" + "".join(f"{r[k]:22.3f}" for k in keys))
