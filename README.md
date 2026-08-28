# NLP-Project — Interpersonal networks from open text

A data science / NLP research project: extract **interpersonal networks** from
open-source text (primary candidate corpus: Wikipedia) and represent them as a graph —
nodes are people, edges are typed social relations.

> *A disagreed with B · C worked with D*

FH-SWF, Angewandte Künstliche Intelligenz (AKI). The deliverable is a paper plus a
reproducible pipeline.


## Where to start

| You are | Read |
|---------|------|
| An AI agent | **[AGENTS.md](AGENTS.md)** — first, always |
| A human contributor | [AGENTS.md](AGENTS.md), then [docs/specs/](docs/specs/) |

## How this repo works

It is **spec-driven**: decisions get written down in [`docs/specs/`](docs/specs/) before
code is written against them. For a research project this means the paper's Methods
section is largely written in advance, and every reported number traces back to a
recorded decision. See [docs/specs/README.md](docs/specs/README.md).

## Pipeline

Seven `inpnet` commands turn a Wikidata query into candidate relation sentences grouped
into field-agnostic clusters. Each stage reads one directory and writes another — no
stage overwrites its own input, so a bug in stage *N* never forces re-running stage
*N − 1* — and every stage writes a `_manifest.json` beside its output recording inputs,
config, tool versions and counts.

```
seed → fetch → clean → resolve → segment → mentions → cluster
└──────── spec 0001 ────────┘└──── spec 0002 ────┘  spec 0003
```

All seven stages have been run on the full corpus.

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
**Writes:** `data/interim/documents.jsonl`.

Strips infoboxes, navboxes, tables, reference lists and similar boilerplate, keeping
prose paragraphs and section headings. Wiki-links are kept as **character offsets** into
the cleaned text rather than inline markup, so downstream stages can treat the text as
plain prose and still recover exactly which span links to which article. **Measured:**
15,158 documents, 47,566 sections, 443,527 wiki links, 53.1 M characters.

### 4. `resolve` — link targets → Wikidata humans

```bash
inpnet resolve --contact you@example.org
```

**Needs:** `requests`. **Reads:** `data/interim/documents.jsonl`. **Writes:**
`data/interim/wikidata_cache.jsonl` (+ `wikidata_cache.jsonl.gz`, tracked in git as the
citable snapshot).

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
NER, no parser). **Reads:** `data/interim/documents.jsonl`. **Writes:**
`data/interim/sentences.jsonl`.

Sentence boundaries are needed to scope candidate pairs (§6) to "mentioned in the same
sentence" rather than "mentioned in the same 10,000-character section." Runs **per
paragraph**, not per section — feeding whole sections in let the model run straight
through blank lines and weld sentences across paragraph breaks. **Measured:** 384,656
sentences.

### 6. `mentions` — person spans + candidate pairs

```bash
inpnet mentions
```

**Needs:** core install only. **Reads:** `documents.jsonl`, `sentences.jsonl`,
`wikidata_cache.jsonl`, `seed.jsonl`. **Writes:** `data/interim/mentions.jsonl`,
`data/interim/entities.jsonl`, `data/interim/candidates.jsonl`.

Two rules generate candidate person-pairs: a linked person paired with the article's own
subject (an article's subject is rarely linked in its own article, so it's added as an
implicit party — the "subject heuristic"), and two linked people sharing a sentence
(capped above 6 people in a sentence, to suppress enumerations like "the seminar was
attended by Bohr, Dirac, Yukawa, …" from generating C(n,2) pairs that assert nothing
between the *listed* people). **Fails hard** if `wikidata_cache.jsonl` doesn't cover the
corpus's link targets, rather than silently producing a near-empty graph (`--lenient` to
downgrade to a warning for deliberate subset runs). **Measured:** 74,149 mentions
(58,991 link + 15,158 subject), 32,546 people (15,158 seed physicists + 17,388 others),
**76,205 candidate person-pairs** — the input to clustering.

### 7. `cluster` — candidate sentences → field-agnostic clusters

```bash
inpnet cluster --cluster-selection-method leaf --n-neighbors 15 --n-components 15 --min-cluster-size 5
```

**Needs:** the `relations` extra (`uv sync --extra relations` — see
[Dependencies](#dependencies)). **Reads:** `data/interim/candidates.jsonl`. **Writes:**
`data/interim/sentence_embeddings.npy`, `sentence_ids.jsonl`, `relation_clusters.jsonl`,
`cluster_summary.jsonl`.

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
