"""Fetch Stack Overflow tag counts (batched) and related-tag co-occurrence for top tags. Cached; quota-aware."""
import json, os, sys, time, requests
API = "https://api.stackexchange.com/2.3"
CACHE = "data/raw/so_cache.json"
cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {"info": {}, "related": {}}

def get(path, **params):
    r = requests.get(f"{API}{path}", params={"site": "stackoverflow", **params}, timeout=30).json()
    if "error_id" in r: sys.exit(f"API error: {r}")
    if r.get("backoff"): time.sleep(r["backoff"])
    print(f"quota_remaining={r.get('quota_remaining')}", file=sys.stderr)
    return r

def save(): json.dump(cache, open(CACHE, "w"))

nodes = json.load(open("data/nodes.json"))
tags = sorted({t for n in nodes for t in n["so_tags"]} - set(cache["info"]))
for i in range(0, len(tags), 100):
    batch = tags[i:i + 100]
    r = get("/tags/" + ";".join(requests.utils.quote(t, safe="") for t in batch) + "/info", pagesize=100)
    for it in r["items"]: cache["info"][it["name"]] = it["count"]
    for t in batch: cache["info"].setdefault(t, 0)
    save(); time.sleep(0.2)

top_n = int(sys.argv[1]) if len(sys.argv) > 1 else 0
top = sorted(cache["info"], key=cache["info"].get, reverse=True)[:top_n]
for t in [t for t in top if t not in cache["related"]]:
    r = get(f"/tags/{requests.utils.quote(t, safe='')}/related", pagesize=100)
    cache["related"][t] = {it["name"]: it["count"] for it in r["items"]}
    save()
    if r.get("quota_remaining", 1) <= 3: sys.exit("quota nearly exhausted; rerun tomorrow")
    time.sleep(0.2)
print(f"info: {len(cache['info'])} tags, related: {len(cache['related'])} tags")
