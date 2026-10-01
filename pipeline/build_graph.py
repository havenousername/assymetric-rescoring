"""Labels → DAG → transitive closure → train/val/test splits. Writes data/graph.json."""
import csv, glob, json, random
import networkx as nx

random.seed(0)
d = json.load(open("data/candidates.json"))
N, C = d["nodes"], d["candidates"]
wiki = json.load(open("data/raw/wiki_summaries.json"))

labels = {}
for f in sorted(glob.glob("data/labels/batch_*.csv")):
    for r in csv.DictReader(open(f)):
        labels[int(r["id"])] = r
missing = [i for i in range(len(C)) if i not in labels]
assert not missing, f"unlabeled ids: {missing[:20]}"

# Same label under several QIDs (Wikidata item + SO-tag item) = one skill; keep the most-asked-about one.
canon = {}
for q in sorted(N, key=lambda q: -N[q]["so_count"]):
    canon.setdefault(N[q]["label"].lower(), q)
C = [{**c, "src": canon[N[c["src"]]["label"].lower()], "dst": canon[N[c["dst"]]["label"].lower()]} for c in C]

G = nx.DiGraph()  # edge = specific → general (Knows src ⊑ Knows dst)
for i, c in enumerate(C):
    if labels[i]["y"] == "y" and c["src"] != c["dst"]:
        G.add_edge(c["src"], c["dst"], conf=labels[i]["conf"], cid=i)
# Mutual implication = near-equivalence (React↔JSX); keeping either direction would be arbitrary.
mutual = [(u, v) for u, v in G.edges if u < v and G.has_edge(v, u)]
G.remove_edges_from(mutual + [(v, u) for u, v in mutual])

# Break cycles: a partial order can't have them. Drop low-confidence edges first.
dropped = []
while True:
    try: cyc = nx.find_cycle(G)
    except nx.NetworkXNoCycle: break
    u, v = min((e[:2] for e in cyc), key=lambda e: G.edges[e]["conf"] == "h")
    dropped.append((N[u]["label"], N[v]["label"])); G.remove_edge(u, v)
G.remove_nodes_from([n for n in list(G) if G.degree(n) == 0])

def closure(g): return {n: nx.descendants(g, n) for n in g}  # ancestors in skill order = graph descendants
full = closure(G)

# Held-out nodes: leaves (nothing more specific below them) that have ≥1 ancestor.
leaves = [n for n in G if G.in_degree(n) == 0 and G.out_degree(n) > 0]
random.shuffle(leaves)
k = len(G) // 10
test_nodes, val_nodes = set(leaves[:k]), set(leaves[k:k + k // 2])
T = G.copy(); T.remove_nodes_from(test_nodes | val_nodes)

# Held-out edges: remove only if the pair becomes unreachable (else transitivity trivially recovers it).
edges = list(T.edges); random.shuffle(edges)
held = {"val": [], "test": []}
target = {"test": len(edges) // 10, "val": len(edges) // 20}
for u, v in edges:
    split = next((s for s in ("test", "val") if len(held[s]) < target[s]), None)
    if split is None: break
    T.remove_edge(u, v)
    if nx.has_path(T, u, v) or T.out_degree(u) == 0:  # keep every train node attached upward
        T.add_edge(u, v); continue
    held[split].append((u, v))
train = closure(T)

def pairs_new(split):
    """Pairs that become implied once this split's held-out edges are added back to the training graph."""
    g = T.copy(); g.add_edges_from(held[split]); c = closure(g)
    return sorted([a, b] for a in T for b in c[a] - train[a])

out = {
    "nodes": {q: {"label": N[q]["label"], "desc": N[q]["desc"], "wiki": wiki.get(q, ""), "so_count": N[q]["so_count"]} for q in G},
    "train_edges": [list(e) for e in T.edges],
    "low_conf_edges": [[u, v] for u, v, c in G.edges(data="conf") if c == "l"],
    "held_edges": {s: [list(e) for e in v] for s, v in held.items()},
    "new_pairs": {s: pairs_new(s) for s in ("val", "test")},
    "heldout_nodes": {"test": sorted(test_nodes), "val": sorted(val_nodes)},
    "full_ancestors": {n: sorted(a) for n, a in full.items()},
    "train_ancestors": {n: sorted(a) for n, a in train.items()},
    "dropped_cycle_edges": dropped,
    "dropped_mutual_pairs": [(N[u]["label"], N[v]["label"]) for u, v in mutual],
}
json.dump(out, open("data/graph.json", "w"), indent=1)
print(f"graph: {len(G)} nodes, {G.number_of_edges()} edges ({len(out['low_conf_edges'])} low-conf), "
      f"{len(mutual)} mutual pairs dropped, {len(dropped)} dropped for longer cycles, closure pairs {sum(map(len, full.values()))}")
print(f"train: {len(T)} nodes, {T.number_of_edges()} edges, closure pairs {sum(map(len, train.values()))}")
print(f"held-out nodes: test {len(test_nodes)}, val {len(val_nodes)}; held-out edges: "
      f"test {len(held['test'])}, val {len(held['val'])}; new pairs test {len(out['new_pairs']['test'])}, val {len(out['new_pairs']['val'])}")
