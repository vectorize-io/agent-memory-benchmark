# boltons real-codebase suite — results

sdebench on a **real codebase** (boltons, 1622 commits): does a local file-based memory (git history +
LLM-summarized developer chats) help a coding agent, against a realistic large store? Dataset design
and reproducibility: see `DATASET.md`.

## Headline (n=5, vanilla vs memsys)

| task | tier | vanilla interv/run | memsys interv/run | vanilla turns | memsys turns |
|---|---|---:|---:|---:|---:|
| under2camel | real-function | 2.0 | 0.0 | 48 | 17 |
| budget | planted | 1.8 | 0.0 | 43 | 14 |
| slugify | real-function | 1.6 | 0.0 | 40 | 27 |
| pluralize | real-function | 1.4 | 0.2 | 42 | 30 |
| parseflag | planted | 1.2 | 0.0 | 30 | 17 |
| discount | planted | 1.0 | 0.0 | 22 | 12 |
| findhashtags | real-function | 1.0 | 0.0 | 28 | 20 |
| listmerge | planted | 1.0 | 0.0 | 26 | 16 |
| rounding | planted | 0.8 | 0.0 | 27 | 14 |
| **TOTAL** | | **56 interventions** | **1 intervention** | **35 avg** | **19 avg (−46%)** |

**Across the 9-task dataset, memory takes the plain agent's 56 human interventions to 1, and cuts
turns 46%** — on a real codebase, with ~1500 real commit subjects as retrieval noise plus decoy chats.
84 runs (42 vanilla + 42 memsys, n=4–5/task); **all solved, 0 legitimacy problems** (source-only
patches, real tests green, no hidden-test leakage).

## Reading it

- **Every task discriminates** — the plain agent needs interventions on all 9 (vanilla mean 0.8–2.0/run),
  because each hides a non-guessable, project-specific decision that lives only in a past chat.
- **memory solves nearly everything at 0** — the one exception is a single `pluralize` run (1 of 5).
- The real-function tasks (`slugify`, `pluralize`, `under2camel`, `findhashtags`) are graded against
  boltons' **real** `test_strutils.py`, so a fix must keep the real library green.

## Quality bar

Two real-function traps were **caught as guessable in the n=1 round** (the plain agent solved them: it
enumerated a broad common-acronym set for `under2camel`; it wrote exactly `not tag.isdigit()` for
`findhashtags`). Both were sharpened to genuinely non-guessable policies — `under2camel` to a
project-specific acronym set `{HTTP,API,SKU,GDPR}` that *excludes* common `db`/`url` (so the broad
guess fails both ways); `findhashtags` to "drop all-numeric **except 4-digit years**" (so the obvious
rule drops `#2024` and fails). This is the difference between a demo and a launch dataset.

## Caveats

- n=4–5 (boltons runs are slow: full-repo copy + Docker grade per run). Single model (gemini-3.5-flash).
- 4 real-function + 5 planted. boltons is very well-tested, so clean non-guessable real-function traps
  are scarce (bytes2human/parse_timedelta/unique were rejected as guessable or messy); 4 is the honest
  strutils ceiling. The planted traps still run inside the real repo (real history noise, real navigation).
- Retrieval is TF-IDF + a code-symbol boost; it holds the right memory in the top-2 across the ~1500-entry
  store, but the margin thins at scale — a much larger store would want embeddings/a reranker.
