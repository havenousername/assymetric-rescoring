"""Merge Wikidata + Stack Overflow co-occurrence into one candidate edge list for labeling."""
import json
from collections import Counter

nodes = {n["qid"]: n for n in json.load(open("data/nodes.json"))}
wd = json.load(open("data/candidate_edges.json")); TL = wd["target_labels"]
so = json.load(open("data/raw/so_cache.json")); info, related = so["info"], so["related"]

# Non-skill categories: "is free software" says nothing about what a person knows.
BLOCK = {"free software", "software", "free and open-source software", "business", "technical standard",
         "open-source software", "computer science term", "application software", "field of study",
         "software category", "proprietary software", "service on Internet", "academic discipline",
         "software as a service", "technique", "software feature", "public company", "enterprise",
         "GNU package", "computer program", "model series", "utility software", "Apache Software Foundation project",
         "Microsoft Windows component", "mathematical concept", "W3C Recommendation", "mobile app", "property",
         "method", "online community", "ISO standard", "concept", "software company", "organization", "quality",
         "identifier", "process", "technology", "specification edition", "online service", "brand", "project",
         "sequence", "social networking service", "file format family", "computer storage media", "certification",
         "free and open-source software", "Python package", "Java software library", "programming tool", "website",
         "type of software", "class of computer programs", "software product"}
# instance_of/subclass_of only count toward categories a project could actually require.
ALLOW = {"web framework", "JavaScript framework", "relational database management system", "database management system",
         "NoSQL database management system", "graph database management system", "key–value database system",
         "cloud database", "query language", "functional programming language", "JVM language", "markup language",
         "cloud computing", "infrastructure as a service", "platform as a service", "container orchestrator",
         "content management system", "build system", "build automation", "continuous integration software",
         "test automation framework", "software testing", "message-oriented middleware", "game engine", "Unix shell",
         "widget toolkit", "natural language processing", "parallel computing", "web server", "application server",
         "hypervisor", "virtual machine", "centralized version control system", "version control system",
         "distributed version control system", "machine learning", "deep learning", "search engine", "Linux distribution",
         "object-relational mapping", "template processor", "static site generator", "front-end framework"}
KEEP = {"programmed_in", "instance_of", "subclass_of", "based_on", "part_of", "depends_on"}

cat_members = Counter(e["dst"] for e in wd["edges"]
                      if e["rel"] in ("instance_of", "subclass_of") and e["dst"] not in nodes)
cats = {q for q in cat_members if TL.get(q) in ALLOW}

def label(q): return (nodes[q]["label"] or nodes[q]["so_tags"][0]) if q in nodes else TL.get(q, q)

cands = {}
for e in wd["edges"]:
    if e["rel"] not in KEEP or label(e["dst"]) in BLOCK: continue
    if e["rel"] in ("instance_of", "subclass_of") and label(e["dst"]) not in ALLOW: continue
    if e["dst"] in nodes or e["dst"] in cats:
        cands.setdefault((e["src"], e["dst"]), {"src": e["src"], "dst": e["dst"], "evidence": []})["evidence"].append(f"wd:{e['rel']}")

tag2q = {t: q for q, n in nodes.items() for t in n["so_tags"]}
for p, rs in related.items():
    for ch, pair in rs.items():
        if ch == p or not info.get(ch) or ch not in tag2q or p not in tag2q: continue
        pc = pair / info[ch]
        if pc >= 0.25 and info[p] >= 2 * info[ch]:
            cands.setdefault((tag2q[ch], tag2q[p]), {"src": tag2q[ch], "dst": tag2q[p], "evidence": []})["evidence"].append(f"so:P(dst|src)={pc:.2f}")

out_nodes = {q: {"qid": q, "label": label(q), "desc": nodes[q]["desc"] if q in nodes else "(category)",
                 "so_count": max((info.get(t, 0) for t in nodes[q]["so_tags"]), default=0) if q in nodes else 0}
             for c in cands.values() for q in (c["src"], c["dst"])}
json.dump({"nodes": out_nodes, "candidates": list(cands.values())}, open("data/candidates.json", "w"), indent=1)
print(len(out_nodes), "nodes,", len(cands), "candidates;", Counter(ev.split(":")[0] + ":" + ev.split(":")[1].split("=")[0] for c in cands.values() for ev in c["evidence"]))
