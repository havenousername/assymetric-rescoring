"""LLM judges: rerank the cos+gen top-K of every test query with a graded p(A→B), and score both
directions of every true test pair. Two judges from different model families: Claude Sonnet 5 (`claude -p`)
and Codex (`codex exec`). The labels were written by Claude, so a Claude judge alone would be circular.
Raw answers cached in data/llm_cache/; metrics merged into results/llm_judge.json.
Usage: uv run python src/llm_judge.py [judge ...]"""
import hashlib, os, re, subprocess, tempfile
from concurrent.futures import ThreadPoolExecutor
from spike import *

K, PER_CALL, PAIRS_PER_CALL, WORKERS = 50, 5, 100, 6
base = cosgen_matrix(0.05)  # λ tuned on val in spike.py; its top-50 holds 99% (A) / 93% (B) of test answers
DEF = ("p(A→B) = probability that a person with solid working experience in skill A has at least basic working "
       "competence in skill B, because working with A requires or exercises B. Judge real-world practice, "
       "not mere co-occurrence.")

def claude(prompt):
    return subprocess.run(["claude", "-p", "--model", "sonnet", "--setting-sources", "", "--tools", "",
                           "--strict-mcp-config", "--no-session-persistence",
                           "--system-prompt", "You are a precise annotator. Output only JSON."],
                          input=prompt, capture_output=True, text=True, timeout=900).stdout

def codex(prompt):
    with tempfile.NamedTemporaryFile("r", suffix=".txt") as f:
        subprocess.run(["codex", "exec", "--skip-git-repo-check", "-s", "read-only", "--ephemeral", "-o", f.name, "-"],
                       input=prompt, capture_output=True, text=True, timeout=900)
        return f.read()

JUDGES = {"claude-sonnet-5": claude, "codex-gpt-6-astra": codex}

def parse(out):
    m = re.search(r"\{.*\}", out or "", re.S)
    try: return json.loads(m.group(0)) if m else None
    except json.JSONDecodeError: return None

def ask(judge, prompt):
    path = f"data/llm_cache/{judge}_{hashlib.sha1(prompt.encode()).hexdigest()[:16]}.txt"
    if os.path.exists(path): return parse(open(path).read())
    for _ in range(3):
        out = JUDGES[judge](prompt)
        if parse(out) is not None:
            open(path, "w").write(out); return parse(out)
    print(f"{judge}: unparseable after 3 tries", file=sys.stderr)

def desc(i, wiki=0):
    n = g["nodes"][ids[i]]
    return f"{n['label']} — {n['desc']}" + (f" ({n['wiki'][:wiki]})" if wiki and n["wiki"] else "")

def rank_prompt(batch):
    lines = [DEF, "", "For each query skill A, score p(A→B) for every candidate B.",
             'Answer with one JSON object only: {"Q1": {"1": p, "2": p, ...}, "Q2": {...}}, p in [0, 1], two decimals.', ""]
    for n, (q, cands) in enumerate(batch, 1):
        lines += [f"Q{n}. A = {desc(q, 300)}"] + [f"  {j}. {desc(c)}" for j, c in enumerate(cands, 1)]
    return "\n".join(lines)

def pair_prompt(batch):
    lines = [DEF, "", 'Score p(A→B) for every numbered pair. Answer with one JSON object only: {"1": p, "2": p, ...}, '
             "p in [0, 1], two decimals.", ""]
    return "\n".join(lines + [f"{j}. A = {desc(a)} | B = {desc(b)}" for j, (a, b) in enumerate(batch, 1)])

def test_tasks():
    """Test queries with their cos+gen top-K, after the same exclusions evaluate() applies; true pairs for direction."""
    qa = [q for q in (ix[q] for q in g["heldout_nodes"]["test"]) if anc_full[q] & set(tr_list)]
    new = {}
    for a, b in g["new_pairs"]["test"]: new.setdefault(ix[a], set()).add(ix[b])
    tasks = []
    for qs, excl in [(qa, lambda q: set()), (sorted(new), lambda q: anc_tr[q] | {q})]:
        S = base(torch.tensor(qs), torch.tensor(tr_list)).cpu()
        for qi, q in enumerate(qs):
            order = [tr_list[j] for j in S[qi].argsort(descending=True).tolist()]
            tasks.append((q, [c for c in order if c not in excl(q)][:K]))
    true = sorted({(q, b) for q in qa for b in anc_full[q] & set(tr_list)} | {(a, b) for a in new for b in new[a]})
    return tasks, true

def judge_all(judge):
    tasks, true = test_tasks()
    both = true + [(b, a) for a, b in true]; random.Random(0).shuffle(both)  # directions never adjacent
    rb = [tasks[i:i + PER_CALL] for i in range(0, len(tasks), PER_CALL)]
    pb = [both[i:i + PAIRS_PER_CALL] for i in range(0, len(both), PAIRS_PER_CALL)]
    with ThreadPoolExecutor(WORKERS) as ex:
        R = list(ex.map(lambda b: ask(judge, rank_prompt(b)), rb))
        P = list(ex.map(lambda b: ask(judge, pair_prompt(b)), pb))
    rank, pair, missing = {}, {}, 0
    for b, r in zip(rb, R):
        for n, (q, cands) in enumerate(b, 1):
            got = (r or {}).get(f"Q{n}") or {}
            for j, c in enumerate(cands, 1):
                try: rank.setdefault(q, {})[c] = float(got[str(j)])
                except (KeyError, TypeError, ValueError): missing += 1
    for b, r in zip(pb, P):
        for j, ab in enumerate(b, 1):
            try: pair[ab] = float((r or {})[str(j)])
            except (KeyError, TypeError, ValueError): missing += 1
    return rank, pair, f"{missing}/{sum(len(c) for _, c in tasks) + len(both)}"

class Judged:
    """Judged top-K first, ordered by cos+gen + p; everything else keeps cos+gen order.
    Ordering by p alone (cos+gen only breaking ties) scored lower on 3 of 4 test MAPs: Claude A 0.779 / B 0.681,
    Codex 0.726 / 0.630 vs 0.773 / 0.698 and 0.757 / 0.686 for the blend."""
    def __init__(s, rank, pair): s.rank, s.pair = rank, pair
    def __call__(s, a, b):
        S, col = base(a, b).clone(), {c: j for j, c in enumerate(b.tolist())}
        for i, q in enumerate(a.tolist()):
            for c, p in s.rank.get(q, {}).items():
                if c in col: S[i, col[c]] += 10 + p  # cos+gen < 1.3, so every judged candidate outranks the rest
        return S
    def pairs(s, a, b): return torch.tensor([s.pair.get(ab, 0.0) for ab in zip(a.tolist(), b.tolist())])

if __name__ == "__main__":
    os.makedirs("data/llm_cache", exist_ok=True)
    try: res = json.load(open("results/llm_judge.json"))
    except FileNotFoundError: res = {}
    for judge in sys.argv[1:] or JUDGES:
        rank, pair, missing = judge_all(judge)
        with torch.no_grad(): res[judge] = evaluate(Judged(rank, pair), "test")
        res[judge]["missing"] = missing
        print(judge, json.dumps(res[judge], default=float), file=sys.stderr)
        json.dump(res, open("results/llm_judge.json", "w"), indent=1, default=float)
