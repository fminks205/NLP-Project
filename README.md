# NLP-Project — Interpersonal networks from open text

A data science / NLP research project: extract **interpersonal networks** from
open-source text (primary candidate corpus: Wikipedia) and represent them as a graph —
nodes are people, edges are typed social relations.

> *A disagreed with B · C worked with D*

**[→ See a rendered example graph](https://htmlpreview.github.io/?https://github.com/fminks205/NLP-Project/blob/main/yale_subset_result.html)**
— an institution-filtered subgraph (people connected by a relation mentioning "Yale"),
rendered via Graphviz with hover tooltips per attribute ([spec 0006](docs/specs/0006-graph-visualization.md)).
GitHub doesn't render `.html` files inline, so this link goes through
[htmlpreview.github.io](https://htmlpreview.github.io) to show it live instead of raw source.

FH-SWF, Angewandte Künstliche Intelligenz (AKI). The deliverable is a paper plus a
reproducible pipeline.


## Where to start

| You are | Read |
|---------|------|
| An AI agent | **[AGENTS.md](AGENTS.md)** — first, always |
| A human contributor | [AGENTS.md](AGENTS.md), then [docs/specs/](docs/specs/) |
| Writing the paper's Methods section | [docs/methodology.md](docs/methodology.md) — a standing digest of pipeline steps, technology, and configuration, kept in sync with the specs and code |

## How this repo works

It is **spec-driven**: decisions get written down in [`docs/specs/`](docs/specs/) before
code is written against them. For a research project this means the paper's Methods
section is largely written in advance, and every reported number traces back to a
recorded decision. See [docs/specs/README.md](docs/specs/README.md).

## Pipeline

`inpnet` commands turn a Wikidata query into candidate relation sentences, each broken
into typed attribute spans and assembled into schemaless relation records. Each stage
reads one directory and writes another — no stage overwrites its own input, so a bug in
stage *N* never forces re-running stage *N − 1* — and every stage writes a
`_manifest.json` beside its output recording inputs, config, tool versions and counts.

```
seed → fetch → clean → resolve → segment → mentions → detect-attributes → assemble-relations
└──────── spec 0001 ────────┘└──── spec 0002 ────┘  └── spec 0004 ──┘   └──── spec 0005 ────┘
```

`cluster` (spec 0003) is **superseded** by `detect-attributes`/`assemble-relations` — see
[spec 0003's Changelog](docs/specs/0003-relation-typology.md) — but its code and command
are kept for reruns/ablation, not removed.

All stages have been run on the full corpus.

All stages that hit a live API (`seed`, `fetch`, `resolve`) require a `--contact` address
or `INPNET_CONTACT` env var — Wikimedia's User-Agent policy requires one — and rate-limit
themselves (`--delay`, default 1 request/second). Every command also accepts `--limit N`
to run on a small subset first.

### 1. `seed` — Wikidata query → the corpus's person list

```bash
inpnet seed --contact you@example.org
```

**Needs:** `requests` (core install). **Reads:** [`docs/queries/physicists.rq`](docs/queries/physicists.rq).
**Writes:** `data/raw/seed.jsonl`.

Runs a SPARQL query against the Wikidata Query Service: humans (`P31 = Q5`) whose
occupation is physicist or a subclass of it, with an English Wikipedia article. Selecting
this way (rather than Wikipedia's category tree) keeps out fictional physicists, films
about physicists, and list articles — see [spec 0001](docs/specs/0001-corpus-acquisition.md).
**Measured:** 15,158 people.

### 2. `fetch` — download each article, pinned to a revision

```bash
inpnet fetch --contact you@example.org
```

**Needs:** `requests`. **Reads:** `data/raw/seed.jsonl`. **Writes:**
`data/raw/html/{qid}.html`, `data/raw/fetch_log.jsonl`.

Fetches Parsoid HTML from the Wikimedia REST API — chosen over raw wikitext or plain-text
extracts because Parsoid renders wiki-links as `<a rel="mw:WikiLink">`, the human-curated
entity-link signal the rest of the pipeline depends on. Each article's revision id is
recorded from the response `ETag`, so the corpus is a citable, reproducible snapshot
rather than "whatever Wikipedia says today." **Measured:** 15,158 articles, 0 errors,
1.2 GB — a multi-hour run at the polite default rate, resumable (`--no-resume` to force
a re-fetch).

### 3. `clean` — HTML → prose with link spans

```bash
inpnet clean
```

**Needs:** `beautifulsoup4`, `lxml`. **Reads:** `data/raw/html/`, `data/raw/seed.jsonl`.
**Writes:** `data/interim/corpus_acquisition/{version}/documents.jsonl`.

Strips infoboxes, navboxes, tables, reference lists and similar boilerplate, keeping
prose paragraphs and section headings. Wiki-links are kept as **character offsets** into
the cleaned text rather than inline markup, so downstream stages can treat the text as
plain prose and still recover exactly which span links to which article. **Measured:**
15,158 documents, 47,566 sections, 443,527 wiki links, 53.1 M characters.

### 4. `resolve` — link targets → Wikidata humans

```bash
inpnet resolve --contact you@example.org
```

**Needs:** `requests`. **Reads:** `data/interim/corpus_acquisition/{version}/documents.jsonl`.
**Writes:** `data/interim/entity_mention_layer/{version}/wikidata_cache.jsonl`
(+ `wikidata_cache.jsonl.gz`, tracked in git as the citable snapshot).

Resolves every distinct link target (title → QID → `P31 = Q5`) via chunked SPARQL, so
downstream stages know which links point at people versus places, institutions, awards,
etc. See [spec 0002](docs/specs/0002-entity-mention-layer.md) for why SPARQL was chosen
over `wbgetentities`. **Measured:** 125,927 distinct link targets resolved; 25,828 are
people.

### 5. `segment` — sections → sentences

```bash
inpnet segment
```

**Needs:** `spacy` + `en_core_web_sm` (core install; only the `senter` pipe runs — no
NER, no parser). **Reads:** `data/interim/corpus_acquisition/{version}/documents.jsonl`.
**Writes:** `data/interim/entity_mention_layer/{version}/sentences.jsonl`.

Sentence boundaries are needed to scope candidate pairs (§6) to "mentioned in the same
sentence" rather than "mentioned in the same 10,000-character section." Runs **per
paragraph**, not per section — feeding whole sections in let the model run straight
through blank lines and weld sentences across paragraph breaks. **Measured:** 384,656
sentences.

### 6. `mentions` — person spans + candidate pairs

```bash
inpnet mentions
```

**Needs:** core install only. **Reads:** `data/interim/corpus_acquisition/{version}/documents.jsonl`,
`data/raw/seed.jsonl`, and from `data/interim/entity_mention_layer/{version}/`:
`sentences.jsonl`, `wikidata_cache.jsonl`. **Writes (same directory):** `mentions.jsonl`,
`entities.jsonl`, `candidates.jsonl`.

Two rules generate candidate person-pairs: a linked person paired with the article's own
subject (an article's subject is rarely linked in its own article, so it's added as an
implicit party — the "subject heuristic"), and two linked people sharing a sentence
(capped above 6 people in a sentence, to suppress enumerations like "the seminar was
attended by Bohr, Dirac, Yukawa, …" from generating C(n,2) pairs that assert nothing
between the *listed* people). **Fails hard** if `wikidata_cache.jsonl` doesn't cover the
corpus's link targets, rather than silently producing a near-empty graph (`--lenient` to
downgrade to a warning for deliberate subset runs). **Measured:** 74,149 mentions
(58,991 link + 15,158 subject), 32,546 people (15,158 seed physicists + 17,388 others),
**76,205 candidate person-pairs** — the shared input to both the superseded `cluster`
stage and to `detect-attributes`/`assemble-relations` below.

### 7. `cluster` — candidate sentences → field-agnostic clusters

```bash
inpnet cluster --cluster-selection-method leaf --n-neighbors 15 --n-components 15 --min-cluster-size 5
```

**Needs:** the `relations` extra (`uv sync --extra relations` — see
[Dependencies](#dependencies)). **Reads:** `data/interim/entity_mention_layer/{version}/candidates.jsonl`
(plus `mentions.jsonl`/`entities.jsonl` for entity masking). **Writes:**
`data/interim/relation_typology/{version}/sentence_embeddings.npy`, `sentence_ids.jsonl`,
`relation_clusters.jsonl`, `cluster_summary.jsonl`.

Deliberately does **not** assign relation labels. The 76,205 candidate pairs collapse to
43,664 distinct sentences (a sentence naming 3+ people yields several pairs); each is
embedded (`sentence-transformers`), dimensionality-reduced (`UMAP`, cosine metric), and
grouped by density (`HDBSCAN`, via `sklearn.cluster.HDBSCAN`) — with no cluster count
fixed upfront, and a native "noise" label for sentences that don't fit any dense group.
This is the field-agnostic layer a domain-specific relation typology (physics-specific
labels, still to be written) plugs into via the contract in
[`docs/specs/typologies/TEMPLATE.yaml`](docs/specs/typologies/TEMPLATE.yaml). See
[spec 0003](docs/specs/0003-relation-typology.md) for the full reasoning, including why a
fixed label set wasn't chosen directly.

**Measured:** 1,286 clusters, 27,881 noise (63.8%), median cluster size 9, run in 4m42s.
HDBSCAN's default `cluster_selection_method="eom"` collapsed the whole corpus into one
cluster (favors the most persistent node in the tree, which at this scale is the root);
`"leaf"` fixed it — see spec 0003's Changelog. Hyperparameters were chosen by
`inpnet cluster-sweep` (below), not guessed.

**`inpnet cluster-sweep`** — a dev tool, not a pipeline stage (nothing it writes lands
under `data/interim/` by default). Embeds a sample once, then cheaply tries a grid of
UMAP/HDBSCAN settings against that one embedding, so comparing hyperparameters costs
seconds per combo instead of minutes. `--show-exemplars N` expands the N most-balanced
combos with real sentences to read, not just cluster-size statistics. See
`src/inpnet/relations/diagnostics.py`.

**Note:** `cluster` (spec 0003) is superseded by `detect-attributes`/`assemble-relations`
below (spec 0003's own clusters kept on cohering around a shared place or topic rather
than a shared relation type, even after person-masking — see
[finding 0002](docs/findings/0002-entity-masking-effect-on-relation-clustering.md) and
spec 0003's Changelog). Kept runnable for reruns/ablation, not removed.

### 8. `detect-attributes` — candidate sentences → typed attribute spans

```bash
inpnet detect-attributes
```

**Reads:** `data/interim/entity_mention_layer/{version}/{candidates,entities,sentences}.jsonl`.
**Writes:** `data/interim/attribute_spans/{version}/attribute_spans.jsonl`.

Runs spaCy's `en_core_web_sm` with `parser`+`ner` enabled (spec 0002/0003 only ever
needed `senter`) over each unique candidate sentence. `time`/`place`/`institution` come
directly from NER labels (`DATE`, `GPE`/`LOC`/`FAC`, `ORG`); `action` is the shortest
dependency-parse path between the two participant tokens of a specific candidate
pair — pair-scoped, not sentence-scoped, since the same sentence can name several pairs
with different actions between them. See
[spec 0004](docs/specs/0004-relation-attribute-detection.md).

**A `0.0.1` full-corpus run found `action` for only 55.7% of pairs.** A diagnostic over
all 33,758 misses traced 95.6% of them to one cause: a `subject_link` candidate whose
subject is referred to only by pronoun in that sentence ("He did doctoral research
under...") — spec 0002's already-documented coreference gap, just never quantified
before (it's **42.4% of every candidate pair in the corpus**). `0.0.2` adds an
accepted-for-now, explicitly flagged heuristic: assume the earliest third-person pronoun
in the sentence refers to the subject (gender-aware where `entities.jsonl`'s `gender`
field is known). This is a real pitfall, not a fix — a sentence can pronoun-reference
someone else entirely — so every affected row/record carries `pronoun_resolved: true`
rather than looking identical to a verified match.

**Measured (`0.0.2`):** 43,664 unique sentences, 76,205 candidate pairs → **154,429
attribute spans** (institution 41,541, time 28,702, place 18,743, action 65,443).
`action` found for **85.9%** of candidate pairs (up from 55.7% in `0.0.1`), of which
**35.2%** (23,010 of 65,443) rest on the pronoun heuristic. Where found, `action` spans
average **94 characters** against a ~187-character average sentence — the enclosing-span
the parse path resolves to is often about half the sentence, broader than a tight verb
phrase like "worked with"; a real, measured characteristic of this heuristic, not a bug.
The remaining ~14% miss is now mostly genuine non-verbal/appositive relations ("his son,
the diplomat...") plus two small, separately-tracked spec 0001/0002 cleaning bugs (see
spec 0004's Open questions) — `0.0.1`'s output is kept for comparison.

### 9. `assemble-relations` — participants + spans → schemaless relation records

```bash
inpnet assemble-relations
```

**Reads:** `candidates.jsonl`, `entities.jsonl`, `sentences.jsonl` (spec 0002),
`attribute_spans.jsonl` (spec 0004). **Writes:**
`data/interim/relations/{version}/relations.jsonl`,
`low_coverage_relations.jsonl`, plus a human-readable `.txt` rendering of each
(`relations_summary.txt`, `low_coverage_relations_summary.txt`) — every relation shown
with its typed attributes inline *and* whichever part of the sentence wasn't captured by
anything, spelled out rather than left implicit in a coverage number.

One record per candidate pair: its two participants plus whatever `time`/`place`/
`institution`/`action` attributes spec 0004 found in its sentence, in an open
`attributes` list rather than fixed columns — a new domain's attribute type is a new
`attr_type` value, not a schema migration. `coverage` (fraction of the sentence's content
tokens actually inside some participant or attribute span) below `--min-coverage`
(default `0.5`) routes a record to `low_coverage_relations.jsonl` instead of
`relations.jsonl`, so a detector gap is visible rather than silently kept or dropped. See
[spec 0005](docs/specs/0005-schemaless-relation-attributes.md).

**Measured (`0.0.2`):** 76,205 candidate pairs → **51,148 relations**, **25,057
low-coverage** (32.9%, down from 50.4% in `0.0.1`, at the `0.5` default threshold — a
proposal to tune once this distribution was actually visible, same posture spec 0003
took with its own hyperparameters), mean coverage 0.624 (up from 0.510). **30.3%**
(23,091 of 76,205) of records carry `pronoun_resolved: true` — visible per-record, not
just as an aggregate, and rendered in `relations_summary.txt` as `[pronoun-resolved
participant -- heuristic, unverified]` so it can't be mistaken for a verified match.
Determinism reconfirmed end to end at `0.0.2` (byte-identical rerun of both stages).

## Dependencies

`uv sync` installs the core install: `requests`, `beautifulsoup4`, `lxml` for corpus
acquisition (spec 0001); `spacy` + `en_core_web_sm` for sentence segmentation (spec
0002); `pyyaml` for the relation-typology plug-in config (spec 0003). All CPU-only, no
model downloads beyond spaCy's small pinned model.

`uv sync --extra relations` additionally installs what spec 0003's `cluster` stage
needs: `sentence-transformers` (sentence embeddings) with a CPU-only `torch` build, plus
`umap-learn` and `scikit-learn` (`sklearn.cluster.HDBSCAN`) for the embed → reduce →
cluster pipeline. This is the project's first ML dependency beyond spaCy's `senter` and
kept optional — a few hundred MB, and only needed once relation clustering is actually
run. `inpnet cluster` fails with a clear message pointing at this command if the extra
isn't installed. No GPU is used or required; see [AGENTS.md](AGENTS.md).

## Ethics

This project makes claims about real, named people. Extracted relations are model output,
not established fact. Source text is CC BY-SA 4.0. See [AGENTS.md](AGENTS.md) §6.
