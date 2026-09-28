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
| 0002 | [Entity & mention layer](docs/specs/0002-entity-mention-layer.md) | Draft | Which spans are people, how they resolve to Wikidata, candidate pairs |
| 0003 | [Relation typology](docs/specs/0003-relation-typology.md) | Superseded (0004, 0005) | Field-agnostic sentence clustering, and the contract a domain-specific relation typology plugs into it through |
| 0004 | [Relation attribute span detection](docs/specs/0004-relation-attribute-detection.md) | Accepted | Detecting `time`/`place`/`institution`/`action` spans per candidate sentence via spaCy NER + dependency parse |
| 0005 | [Schemaless relation attributes](docs/specs/0005-schemaless-relation-attributes.md) | Accepted | Assembling participants + detected attributes into an open, extensible relation record; low-coverage records split out for review |
| 0006 | [Graph visualization](docs/specs/0006-graph-visualization.md) | Draft | Rendering a Graphviz subgraph filtered by institution substring, edges annotated from the relation record's attribute bag |

## 4. Repository layout

Directories marked *(planned)* do not exist yet — create them when work actually calls
for them, not preemptively.

```
NLP-Project/
├── AGENTS.md                  # you are here
├── CLAUDE.md                  # pointer to this file
├── README.md                  # human-facing intro
├── pyproject.toml             # uv-managed; `uv sync` then `uv run inpnet ...`
├── data_versions.json          # current version per data/interim/ layer — bump by hand
├── docs/
│   ├── specs/                 # ← all specs live here
│   │   └── typologies/        #   relation-typology plug-in configs (spec 0003)
│   ├── queries/                # SPARQL used by the pipeline, version-controlled
│   ├── methodology.md          # digest for the paper's Methods section — keep current, see §5
│   ├── findings/                # dated notes from real runs, feeds the paper's Results/Discussion
│   └── paper/                  # (planned) source of the paper
├── src/inpnet/                # the package — importable, tested code
│   ├── wiki.py                #   Wikimedia API client (spec 0001)
│   ├── manifest.py            #   run manifests + JSONL helpers
│   ├── cli.py                 #   `inpnet` entry point
│   ├── corpus/                #   seed / estimate / fetch / clean  (spec 0001)
│   ├── nlp/                   #   resolve / segment / mentions     (spec 0002)
│   └── relations/             #   attributes / assemble (0004, 0005); cluster / typology
│                               #   (spec 0003, superseded — kept for reruns/ablation)
├── notebooks/                 # (planned) exploration only — never imported by src/
├── tests/                     # pytest
└── data/                      # gitignored, except data/annotations/
    ├── raw/                   #   immutable downloads
    ├── interim/               #   intermediate artifacts, versioned per layer — see below
    │   ├── corpus_acquisition/{version}/      #   spec 0001 output
    │   ├── entity_mention_layer/{version}/    #   spec 0002 output
    │   ├── relation_typology/{version}/       #   spec 0003 output (superseded, kept)
    │   ├── attribute_spans/{version}/         #   spec 0004 output
    │   └── relations/{version}/               #   spec 0005 output
    ├── processed/             #   final outputs for the paper
    └── annotations/           #   hand-labelled data — TRACKED in git
```

**Data rules:**

- `data/raw/` is **immutable**. Never edit in place; re-derive instead.
- Each stage reads from one directory and writes to another. No stage overwrites its
  own input.
- Everything under `data/` is gitignored **except** `data/annotations/` — hand-made
  labels are expensive and belong in version control.
- **`data/interim/` is versioned per layer**, one directory per pipeline layer
  (`corpus_acquisition`, `entity_mention_layer`, `relation_typology`), each holding
  `{version}/` subdirectories so a rerun with different config never silently overwrites
  an existing snapshot — and per-stage `_manifest.json` files stop colliding when several
  stages used to share one flat directory. [`data_versions.json`](data_versions.json) at
  the repo root records each layer's *current* version and is **incremented by hand** —
  there is no automatic bump. `src/inpnet/versions.py` reads it; a stage resolves its own
  output directory from its layer's current version, and resolves its input from the
  version of whichever layer produced it.

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
- **Methodology digest**: [`docs/methodology.md`](docs/methodology.md) is the standing,
  human-facing digest of pipeline steps, technology, and actual configuration — written
  so a human can draft the paper's Methods section from it without re-reading every spec
  and source file. It is derived from the specs and the code, not a second place decisions
  get made; if it and a spec disagree, the spec wins and the digest is stale. **Update it
  in the same commit as any change that would change what it says** — a new/changed
  pipeline stage, a changed CLI default or hyperparameter, a spec moving `Draft →
  Accepted` or `Accepted → Implemented`, or a new full-corpus run with different measured
  numbers. Same discipline as keeping a spec's `Status:` current (§2).

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
