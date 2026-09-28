"""Parse raw QLever TSVs into data/nodes.json and data/candidate_edges.json."""
import csv, json, re
from collections import defaultdict

P = {"P31": "instance_of", "P279": "subclass_of", "P277": "programmed_in", "P1547": "depends_on",
     "P144": "based_on", "P361": "part_of", "P400": "platform", "P306": "operating_system"}

def qid(s): return s.rsplit("/", 1)[-1].rstrip(">")
def lit(s): m = re.match(r'"(.*)"@en$', s or ""); return m.group(1) if m else ""

nodes = {}
for r in csv.reader(open("data/raw/items.tsv"), delimiter="\t", quoting=csv.QUOTE_NONE):
    if r[0].startswith("?") or "stackoverflow.com/tags/" not in r[1]: continue
    q = qid(r[0])
    n = nodes.setdefault(q, {"qid": q, "label": lit(r[2]), "desc": lit(r[3]),
                             "sitelinks": int(r[4]) if r[4].isdigit() else 0, "so_tags": []})
    t = r[1].split("/tags/")[1].rstrip(">")
    if t not in n["so_tags"]: n["so_tags"].append(t)

edges, tlabels = [], {}
for r in csv.reader(open("data/raw/edges.tsv"), delimiter="\t", quoting=csv.QUOTE_NONE):
    if r[0].startswith("?"): continue
    s, t = qid(r[0]), qid(r[2])
    if s not in nodes or s == t: continue
    tlabels[t] = lit(r[3])
    edges.append({"src": s, "rel": P[qid(r[1])], "dst": t})
edges = [dict(e) for e in {tuple(e.items()) for e in edges}]
json.dump(list(nodes.values()), open("data/nodes.json", "w"), indent=1)
json.dump({"edges": edges, "target_labels": tlabels}, open("data/candidate_edges.json", "w"), indent=1)
inside = sum(e["dst"] in nodes for e in edges)
print(f"{len(nodes)} nodes, {len(edges)} edges, {inside} with target also an SO-tagged node")
