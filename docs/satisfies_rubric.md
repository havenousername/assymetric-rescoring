# `satisfies(A → B)` labeling rubric

Meaning: someone with solid working experience in **A** can be assumed to have at least basic
working competence in **B**, *because working with A requires working with B*.
In box terms: `box(Knows A) ⊆ box(Knows B)`.

## YES (`y`)
1. **API language** — A is used by writing code in B. Spring Boot→Java, React→JavaScript, pandas→Python.
2. **Kind of skill category** — A is a kind of B and B is something a project could require.
   PostgreSQL→relational DBMS, Haskell→functional programming language, AWS→cloud computing.
3. **Required layer** — using A means operating B directly. Spring Boot→Spring Framework,
   Sass→CSS, TypeScript→JavaScript, Next.js→React, JDBC→Java.
4. **Component of** — A is a feature/sub-technology of B, so knowing A means working in B. Flexbox→CSS.

## NO (`n`) + reason code
- `impl` — B is only the hidden implementation language. PostgreSQL→C, Docker→Go, NumPy→Fortran, TensorFlow→C++.
- `cooc` — commonly used together, not required. MySQL→PHP, HTML→JavaScript, Firebase→Android.
- `dep` — transitive package dependency users never touch. TensorFlow→six, Django→pytz.
- `hist` — lineage/predecessor, not required. PostgreSQL→POSTGRES.
- `rev` — the implication runs the other way (B ⇒ A).
- `nonskill` — A or B is not a technical skill (natural language, company, consumer product, vague concept).
- `other` — anything else.

## Confidence
`h` = clear case; `l` = defensible but arguable. Low-confidence YES edges are kept but flagged for audit.
