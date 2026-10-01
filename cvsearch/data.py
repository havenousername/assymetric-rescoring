"""Djinni English CVs (lang-uk/recruitment-dataset-candidate-profiles-english, MIT; Drushchak & Romanyshyn 2024) and
keywords mined from the CVs themselves. Nothing here reads the skill graph in data/graph.json.

Splits are by a hash of the CV id: 80% "graph" (vocabulary and graph construction), 5% val, 15% test. Val and test
keep the CVs whose Primary Keyword is one of TECH, at most CAP per keyword. The Primary Keyword is the candidate's own
choice on the platform, separate from the text; it is ground truth for evaluation only and never used to build anything.

Keywords: CVs list their skills ("Strong C, C++, bash, gcc, llvm, clang"). An item counts as listed only inside a run of
at least three short items, which prose almost never produces; listed in at least min_df CVs, it becomes a keyword.
Mining reads Moreinfo and Highlights only: Position is a job title ("Java Developer"), not a list. Extraction reads all
three, every span: "Java Developer" names both Java Developer and Java. A one-word keyword written with spaces names only
itself: "Java Script" is JavaScript, not also Java. Variants merge on a key without dots,
hyphens and spaces (react.js = reactjs = react js). Keywords of one or two letters (C, R, Go) match only when not written
in lower case. A keyword spelled in lower case ("using", "currently") must be listed in at least min_list_share of the CVs
that use it: those are mostly prose words. A version suffix folds into its keyword when that keyword exists and has at
least three characters: C++17, C++ 17 → C++; Python3 → Python; .NET 6 → .NET (ES6, S3, EC2 stay as they are).
"""
import re
from collections import Counter
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parent.parent
PARQUET = ROOT / "data" / "djinni" / "cvs.parquet"
URL = "https://huggingface.co/api/datasets/lang-uk/recruitment-dataset-candidate-profiles-english/parquet/default/train/0.parquet"
TECH = ["JavaScript", "Java", ".NET", "Python", "PHP", "Node.js", "DevOps", "iOS", "Android", "C++", "Unity", "Ruby",
        "Golang", "SQL", "Flutter", "Scala", "Rust", "Salesforce"]
CAP = {"val": 200, "test": 600}
BODY = "coalesce(Moreinfo, '') || chr(10) || coalesce(Highlights, '')"  # "Looking For" holds wishes (friendly team)
TEXT = f"coalesce(Position, '') || chr(10) || {BODY}"
BUCKET = "hash_bucket"
SQL_BUCKET = "cast(('0x' || substr(md5(id), 1, 8))::int64 % 100 as int)"

TOKEN = re.compile(r"[\w+#.&'’-]+")
LIST_SPLIT = re.compile(r"[\n,;|•·●▪◦/]+")
LEADING = re.compile(r"^(?:[\s\-–—*>]+|\d+[.)]\s+)+")
VERSION = re.compile(r"(.{3,}?)v?\d+")
FUNCTION_WORDS = {"and", "or", "with", "the", "a", "an", "also", "etc", "including", "e.g", "i.e", "such", "like", "of",
                  "in", "on", "for", "to", "my", "i", "we", "as", "at", "by", "from", "is", "are", "be"}


def download():
    if not PARQUET.exists():
        import urllib.request
        PARQUET.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(URL, PARQUET)
    return PARQUET


def _query(sql, args=()):
    con = duckdb.connect()
    con.execute(f"create view cv as select *, {SQL_BUCKET} as {BUCKET}, {TEXT} as text, {BODY} as body from '{download()}'")
    return con.execute(sql, args)


def graph_cvs():
    """(body, full text) of every CV in the graph split, all Primary Keywords."""
    return _query(f"select body, text from cv where {BUCKET} >= 20").fetchall()


def eval_cvs(split):
    """[{id, position, text, pk, years}] for val (buckets 0–4) or test (5–19), TECH keywords only, CAP per keyword."""
    lo, hi = (0, 5) if split == "val" else (5, 20)
    rows = _query(f"""select id, Position, text, "Primary Keyword", "Experience Years" from cv
        where {BUCKET} >= ? and {BUCKET} < ? and "Primary Keyword" in ({",".join("?" * len(TECH))})
        qualify row_number() over (partition by "Primary Keyword" order by md5(id)) <= ? order by id""",
                  [lo, hi, *TECH, CAP[split]]).fetchall()
    return [{"id": i, "position": p, "text": t, "pk": k, "years": y} for i, p, t, k, y in rows]


def tokens(text):
    """Word tokens keeping + # . & (C++, C#, .NET, node.js, R&D), trailing dots and quotes cut."""
    return [t.rstrip(".'’-") for t in TOKEN.findall(text) if t.rstrip(".'’-")]


def key(toks):
    return re.sub(r"[.\-]", "", "".join(toks)).lower()


def _item(seg):
    """A short list item: 1–3 tokens, ≤ 30 characters, a letter, no function word up front."""
    seg = LEADING.sub("", seg.split(":")[-1]).strip().strip("!?\"'").strip()
    toks = tokens(seg)
    ok = 0 < len(seg) <= 30 and 1 <= len(toks) <= 3 and re.search(r"[a-zA-Z]", seg) and toks[0].lower() not in FUNCTION_WORDS
    return toks if ok else None


def list_items(text, min_run=3):
    """Items in runs of at least min_run consecutive short items of comma/line/bullet lists."""
    run = []
    for seg in LIST_SPLIT.split(re.sub(r"\([^)]{0,60}\)", " ", text)) + [""]:
        if item := _item(seg):
            run.append(item)
            continue
        if len(run) >= min_run: yield from run
        run = []


def ngrams(text, n=3):
    """(key, surface) for every 1–3 token span. Short keys keep their case for the lower-case rule."""
    toks = tokens(text)
    for i in range(len(toks)):
        for j in range(i + 1, min(i + n, len(toks)) + 1):
            yield key(toks[i:j]), toks[i:j]


class Vocabulary:
    """Keywords mined from list items: keys listed by at least min_df CVs, shown in their most common spelling."""

    def __init__(self, keys, surface, df, variants=None):
        self.keys, self.surface, self.df = keys, surface, df
        self.index = {k: i for i, k in enumerate(keys)}
        self.variants = variants or {}  # version spelling → its keyword's key
        self.index.update({v: self.index[k] for v, k in self.variants.items()})
        self.one_word = {i for i, sf in enumerate(surface) if len(tokens(sf)) == 1}

    @classmethod
    def mine(cls, bodies, min_df=5, min_list_share=0.2):
        bodies = list(bodies)
        listed, forms = Counter(), Counter()
        for t in bodies:
            seen = {}
            for toks in list_items(t): seen.setdefault(key(toks), " ".join(toks))
            listed.update(seen.keys())
            forms.update(seen.items())
        best = {}
        for (k, s), n in forms.items():
            if listed[k] >= min_df and n > best.get(k, ("", 0))[1]: best[k] = (s, n)
        lower = {k for k, (s, _) in best.items() if s.islower()}
        named = Counter()
        for t in bodies: named.update({k for k, _ in ngrams(t) if k in lower})
        keep = {k for k in best if k not in lower or listed[k] >= min_list_share * named[k]}
        variants = {k: m.group(1) for k in keep if (m := VERSION.fullmatch(k)) and m.group(1) in keep}
        base = sorted(keep - set(variants))
        return cls(base, [best[k][0] for k in base], [listed[k] for k in base], variants)

    def _spans(self, toks):
        """(start, end, id) of every keyword span, with the lower-case rule for 1–2 letter keys."""
        out = []
        for i in range(len(toks)):
            for j in range(i + 1, min(i + 3, len(toks)) + 1):
                k = key(toks[i:j])
                if (kid := self.index.get(k)) is not None and (len(k) > 2 or not all(t.islower() for t in toks[i:j])):
                    out.append((i, j, kid))
        return out

    def extract(self, text):
        """Keyword ids named in the text (every span), in order of first mention. Spans inside a spaced one-word
        keyword ("Java Script") don't count."""
        spans = self._spans(tokens(text))
        spaced = [(i, j) for i, j, k in spans if j - i > 1 and k in self.one_word]
        out = {}
        for i, j, k in spans:
            if not any(a <= i and j <= b and (i, j) != (a, b) for a, b in spaced): out.setdefault(k, None)
        return list(out)

    def requirements(self, text):
        """Keyword ids a request names, longest match first: "Spring Boot developer" is Spring Boot, not also Spring."""
        spans, out, end = sorted(self._spans(tokens(text)), key=lambda x: (x[0], x[0] - x[1])), {}, 0
        for i, j, k in spans:
            if i >= end: out.setdefault(k, None); end = j
        return list(out)

    def contained(self):
        """{(a, b)}: keyword b is a span of keyword a's own name (Unreal Engine ⊃ Engine). They co-occur by construction."""
        out = set()
        for a, s in enumerate(self.surface):
            toks = tokens(s)
            for i in range(len(toks)):
                for j in range(i + 1, len(toks) + 1):
                    b = self.index.get(key(toks[i:j]))
                    if b is not None and b != a: out.add((a, b))
        return out

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, i):
        return self.surface[i]

    def find(self, name):
        return self.index[key(tokens(name))]
