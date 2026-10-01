"""Self-check on toy CVs, no download and no model: uv run python -m cvsearch.check"""
import torch

from .data import Vocabulary
from .graph import Cooccurrence, KeywordGraph, subsumption_edges
from .models import graph_lookup
from .search import CVIndex, Search

LISTS = ["C++, Qt, STL, LLVM", "C++, Qt, STL, Linux", "C++, STL, Boost, C++17", "Java, Spring Boot, Hibernate",
         "Java, Spring Boot, Maven", "Java, Maven, Git", "Python, Django, Git", "Python, pandas, NumPy", "JavaScript, HTML, CSS"] * 3
BODIES = LISTS + ["Currently, I work at a bank and go to the gym every day"] * 5

if __name__ == "__main__":
    V = Vocabulary.mine(BODIES, min_df=2)
    names = {V[i] for i in range(len(V))}
    assert {"C++", "Qt", "LLVM", "Spring Boot", "Java"} <= names, names
    assert "Currently" not in names, "prose isn't a list"
    assert V.find("C++ 17") == V.find("C++17") == V.find("c++"), "version folds into its keyword"
    assert V.extract("C++ developer, Java") == [V.find("C++"), V.find("Java")]
    assert V.find("Java") in V.extract("Spring Boot and Java"), "every span"
    assert V.find("Java") not in V.extract("PHP, Java Script, HTML"), "a spaced one-word keyword names only itself"
    assert V.find("JavaScript") in V.extract("PHP, Java Script, HTML")
    assert V.requirements("Need a Spring Boot developer, Java") == [V.find("Spring Boot"), V.find("Java")], "longest match"

    sets = [V.extract(t) for t in LISTS]
    co = Cooccurrence.count(sets, len(V))
    E = {(V[a], V[b]) for a, b, _ in subsumption_edges(co, V.contained(), min_lift=1.0, min_co=2)}
    assert ("LLVM", "C++") in E and ("Spring Boot", "Java") in E, E
    assert ("C++", "LLVM") not in E, "edges point to the more common keyword"
    g = KeywordGraph(V, [(V.find(a), V.find(b), 1.0) for a, b in E], torch.randn(len(V), 8))
    assert V.find("C++") in g.ancestors_train[V.find("LLVM")]

    cvs = [{"text": t, "pk": "", "position": ""} for t in ["Qt, STL", "LLVM and Clang", "C++ only", "Python, Django"]]
    search = Search(CVIndex(cvs, V), {"graph": graph_lookup(g)})
    req, rows = search.query("Looking for the C++ experts", "down", "graph", k=4)
    assert req == ["C++"] and rows[0]["cv"]["text"] == "C++ only", "a CV naming the requirement comes first"
    assert rows[-1]["cv"]["text"] == "Python, Django", "an unrelated CV comes last"
    req, rows = search.query("Would be nice if has LLVM knowledge", "up", "graph", k=4)
    assert rows[0]["cv"]["text"] == "LLVM and Clang", "a CV naming the requirement comes first"
    assert {r["cv"]["text"] for r in rows[1:3]} == {"Qt, STL", "C++ only"}, "then CVs with what LLVM implies (C++, Qt, STL)"
    assert rows[-1]["cv"]["text"] == "Python, Django"
    print("cvsearch check passed")
