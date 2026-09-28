"""Fetch Wikipedia lead extracts for nodes via the batch action API (20 titles/request) → data/raw/wiki_summaries.json."""
import csv, json, os, re, time, requests
OUT = "data/raw/wiki_summaries.json"
done = json.load(open(OUT)) if os.path.exists(OUT) else {}
titles = {}
for r in csv.reader(open("data/raw/enwiki_titles.tsv"), delimiter="\t", quoting=csv.QUOTE_NONE):
    if r[0].startswith("?"): continue
    titles[re.match(r'"(.*)"@en', r[1]).group(1)] = r[0].rsplit("/", 1)[-1].rstrip(">")
H = {"User-Agent": "asymmetric-skills-research/0.1 (research prototype)"}
todo = [t for t, q in titles.items() if not done.get(q)]
for i in range(0, len(todo), 20):
    batch = todo[i:i + 20]
    for wait in (0, 5, 20, 60):
        time.sleep(wait)
        r = requests.get("https://en.wikipedia.org/w/api.php", headers=H, timeout=60, params={
            "action": "query", "prop": "extracts", "exintro": 1, "explaintext": 1, "exlimit": 20,
            "redirects": 1, "format": "json", "titles": "|".join(batch)})
        if r.status_code == 200: break
    data = r.json()["query"]
    alias = {x["to"]: x["from"] for x in data.get("normalized", []) + data.get("redirects", [])}
    for p in data["pages"].values():
        t = p.get("title", ""); t = alias.get(t, t); t = alias.get(t, t)
        if t in titles: done[titles[t]] = p.get("extract", "")
    json.dump(done, open(OUT, "w"), indent=1)
    time.sleep(0.5)
print(len(done), "summaries,", sum(bool(v) for v in done.values()), "non-empty of", len(titles))
