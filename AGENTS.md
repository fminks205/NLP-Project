# AGENTS.md — Entry point for AI agents

**Read this file first.** It tells you what this project is, where things live, and the
one rule that governs how work happens here: **specs before code**.

---

## 1. What this project is

A **data science / NLP research project**. The deliverable is a **paper plus a
reproducible pipeline**.

**Research goal:** extract **interpersonal networks** from open-source text (primary
candidate corpus: Wikipedia) and represent them as a graph — nodes are people, edges are
typed social relations.

> "A disagreed with B", "C worked with D"

Course context: FH-SWF, Angewandte Künstliche Intelligenz (AKI).

**Everything beyond that paragraph is undecided.** Corpus, relation types, models,
evaluation design — none of it is settled. Do not assume; check `docs/specs/`, and if the
answer isn't there, ask.

## 2. The one workflow rule

This repo is **spec-driven**. Every non-trivial decision is written down in
[`docs/specs/`](docs/specs/) before code implements it.

```
idea → spec (Draft) → review → spec (Accepted) → code + tests → spec (Implemented)
```

**Before writing code, an agent must:**

1. Read [`docs/specs/README.md`](docs/specs/README.md) — the spec process.
2. Read the spec covering the area being touched (index below).
3. If **no spec covers the work**, write one from
   [`docs/specs/TEMPLATE.md`](docs/specs/TEMPLATE.md) and get it accepted **before**
   implementing. Small obvious fixes (typos, lint, a failing test) are exempt.
4. If the code would **contradict** an accepted spec, stop and say so. Do not silently
   diverge — either the code is wrong or the spec is, and that is the humans' call.

An agent may write and revise **drafts** freely. An agent may **not** promote a spec to
`Accepted` — only a human does that.

When implementing a spec, update its `Status:` and acceptance checkboxes in the same
commit as the code.

## 3. Spec index

Add a row here in the same commit as any new spec.

| ID | Spec | Status | What it decides |
|----|------|--------|-----------------|
| 0001 | [Corpus acquisition](docs/specs/0001-corpus-acquisition.md) | Draft | Which Wikipedia articles, how they're selected, fetched, and cleaned |

## 4. Repository layout

Directories marked *(planned)* do not exist yet — create them when work actually calls
for them, not preemptively.

```
NLP-Project/
├── AGENTS.md                  # you are here
├── CLAUDE.md                  # pointer to this file
├── README.md                  # human-facing intro
├── pyproject.toml             # uv-managed; `uv sync` then `uv run inpnet ...`
├── docs/
│   ├── specs/                 # ← all specs live here
│   ├── queries/               # SPARQL used by the pipeline, version-controlled
│   └── paper/                 # (planned) source of the paper
├── src/inpnet/                # the package — importable, tested code
│   ├── wiki.py                #   Wikimedia API client (spec 0001)
│   ├── manifest.py            #   run manifests + JSONL helpers
│   ├── cli.py                 #   `inpnet` entry point
│   └── corpus/                #   seed / estimate / fetch / clean  (spec 0001)
├── notebooks/                 # (planned) exploration only — never imported by src/
├── tests/                     # pytest
└── data/                      # gitignored, except data/annotations/
    ├── raw/                   #   immutable downloads
    ├── interim/               #   intermediate artifacts
    ├── processed/             #   final outputs for the paper
    └── annotations/           #   hand-labelled data — TRACKED in git
```

**Data rules:**

- `data/raw/` is **immutable**. Never edit in place; re-derive instead.
- Each stage reads from one directory and writes to another. No stage overwrites its
  own input.
- Everything under `data/` is gitignored **except** `data/annotations/` — hand-made
  labels are expensive and belong in version control.

## 5. Conventions

- **Python 3.13, managed by [uv](https://docs.astral.sh/uv/).** `uv sync` to install,
  `uv run inpnet ...` to run, `uv run pytest` to test. There is no system Python on this
  machine — `python` resolves to the Windows Store stub, so always go through `uv run`.
- Package lives under `src/inpnet/`, tests under `tests/`.
- **Determinism**: seed everything. A rerun must reproduce the numbers in the paper.
- **Provenance**: any artifact derived from a source document keeps a traceable link back
  to that document and its version.
- **Config over constants**: no hardcoded paths or magic thresholds buried in `src/`.
- **Manifests**: every stage writes `_manifest.json` beside its output — inputs, hashes,
  config, versions, counts.
- **Docstrings** on public functions name the spec they implement.
- **Notebooks** are for exploration and figures. Logic that matters moves into `src/`.
- **Commits** reference the spec they implement, e.g. `spec 0003: ...`.

**Network etiquette is a hard requirement, not a nicety.** Wikimedia's User-Agent policy
requires an identifiable contact; `WikiClient` refuses to construct without one. Pass
`--contact you@example.org` or set `INPNET_CONTACT`. Never remove the rate limiting.

## 6. Ethics & licensing

The output contains **claims about real, named people**, some living. That constrains the
work:

- Only **public, openly licensed** sources. Wikipedia text is CC BY-SA 4.0 — attribution
  and share-alike apply to derived data we publish.
- Extracted relations are **model output, not fact**. Any published artifact must label
  them as such and report accuracy honestly.
- Prefer keeping evidence and source references attached to extracted claims, so nothing
  in the output is an unsourced assertion. (How this is enforced technically → spec.)
- Wikipedia's coverage is skewed by gender, geography, and era. Any network built from it
  inherits that, and the paper should say so.

## 7. Working notes for agents

- **Don't invent numbers or decisions.** If it wasn't run, don't report it. If it wasn't
  decided, don't write it into a spec as though it were — put it under *Open questions*
  and ask.
- **Check the actual state of the repo** before claiming anything works. This project is
  at the scaffolding stage.
- Prefer extending an existing spec over creating a near-duplicate.
