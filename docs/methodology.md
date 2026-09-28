# Methodology digest

| | |
|---|---|
| **Purpose** | The factual substrate for the paper's Methods section — every technology, configuration, and measured number a human needs to *write* the prose, without re-reading every spec and source file. |
| **Not** | A second place decisions get made. Every fact here traces to an accepted/draft spec or a source file, cited inline. If this file and a spec disagree, **the spec is right and this file is stale** — fix the file. |
| **Update discipline** | Same commit as any change that changes what this says: a new/changed pipeline stage, a changed CLI default or hyperparameter, a spec moving `Draft → Accepted` / `Accepted → Implemented`, or a new full-corpus run with different measured numbers. See [AGENTS.md](../AGENTS.md) §5. |
| **Last synced against** | commit `c10f568` (2026-08-29) |

---

## 1. Overview

**Research goal.** Extract interpersonal networks (who worked with, disagreed with,
taught whom) from English Wikipedia biographies into a typed, evidence-bearing graph:
nodes are people, edges are social relations grounded in the sentence that asserts them.

**First-pass domain.** Physicists on English Wikipedia — a densely interconnected,
well-documented community, chosen as a first corpus, not a permanent scope restriction.
The pipeline itself is domain-agnostic through the mention layer; physics-specific
relation labels are still a not-yet-written follow-on spec, now plugging into
`assemble-relations`'s `action` attribute (§5) rather than whole-sentence clusters.

**Pipeline.** Nine `inpnet` CLI commands, each its own stage: reads one directory, writes
another, never overwrites its own input. Every stage writes a `_manifest.json` beside its
output (input file hashes, config, tool/model versions, timestamp, counts) — the
mechanism that makes every number below re-derivable rather than asserted.

```
seed → fetch → clean → resolve → segment → mentions → detect-attributes → assemble-relations
└──────── corpus acquisition ────┘└─── entity & mention layer ──┘  └── attributes ──┘  └── relations ──┘
              (spec 0001)                    (spec 0002)             (spec 0004)          (spec 0005)
```

`cluster` (spec 0003) is superseded by the last two stages (§5) but kept runnable for
reruns/ablation. All nine commands have been run end-to-end on the full corpus at least
once. Spec statuses as of the sync date above: 0001 *Draft*, 0002 *Draft*, 0003
*Superseded*, 0004/0005 *Accepted* (both implemented and run; determinism check and a
more formal review of `low_coverage_relations.jsonl` still open). "Draft" for 0001/0002
means implemented-and-run but not yet signed off by a human as the final decision — see
each spec's Open Questions for what's still unsettled.

**Environment.** Python 3.13, dependency/build management via `uv`. No GPU (16 CPU cores
available at time of writing); every model choice through spec 0003 is explicitly
CPU-feasible for that reason, and specs 0004/0005 kept that constraint — `parser`/`ner`
are the same pinned `en_core_web_sm` model spec 0002 already depends on, not a new or
GPU-dependent model. Coreference and zero-shot RE remain deferred and out of scope.

---

## 2. Corpus acquisition (spec [0001](specs/0001-corpus-acquisition.md))

### 2.1 Selection

Seed set defined by one version-controlled SPARQL query,
[`docs/queries/physicists.rq`](queries/physicists.rq), against the Wikidata Query
Service (WDQS): humans (`wdt:P31 wd:Q5`) whose occupation (`wdt:P106`) is physicist or a
transitive subclass of it (`wdt:P279*`), restricted to those with an English Wikipedia
article.

Chosen over recursing Wikipedia's `Category:Physicists` tree, which is contaminated with
non-person categories (fictional physicists, films about physicists, list articles) and
has no safe recursion depth. The trade-off to state in the paper: this selection method
**silently misses** people English Wikipedia categorizes as physicists but whose
Wikidata item lacks `P106`.

`SERVICE wikibase:label` is deliberately not used (pushed the query past WDQS's timeout;
WDQS reports that timeout as HTTP 200 with a truncated stream, which the client detects
and raises on). Display names come from the article title instead.

**Measured (2026-08-12):**

| Selection | Count |
|---|---|
| Strict `P106 = Q169470` | 11,971 |
| With subclass traversal (`P106/P279*`) | **15,158** (used) |

### 2.2 Retrieval

Wikimedia REST API, `GET /api/rest_v1/page/html/{title}` — one request per article. The
response is **Parsoid HTML**, and its `ETag` header carries the revision id
(`W/"{revision_id}/…"`), so each article is pinned to a specific revision at fetch time
with no second request needed.

**Why Parsoid HTML, not raw wikitext or plain-text extracts:** Parsoid renders wiki-links
as `<a rel="mw:WikiLink" href="./Page_Title">` — a human-curated entity-link signal
(someone deliberately linked "his wife" to `./Marie_Curie`) that plain-text extraction
(`prop=extracts`, the HuggingFace `wikimedia/wikipedia` dataset) discards. It also
resolves templates, which raw wikitext would leave to interpret. This is the corpus's
central design choice and should be foregrounded in the paper.

**Etiquette / reproducibility constraints, both worth stating in Methods:**
- Descriptive `User-Agent` with a contact address, serial requests, ~1 req/s, retry with
  backoff honoring `Retry-After` — Wikimedia's policy, not an optional nicety.
- Revision pinning means a rerun months later reproduces the same snapshot rather than
  drifting with live edits.

**Measured:** 15,158 articles fetched, 0 errors, 1.2 GB raw HTML (matched the 1.2 GB
projection from a 40-article HTML:wikitext size-ratio sample, 7.87×). Wikitext total
(exact, via `prop=info`): 151.6 MB; mean/median/max article size 10.2 KB / 7.1 KB / 269 KB.

### 2.3 Cleaning

HTML → prose, keeping paragraphs and section headings (section context such as "Later
life" feeds downstream features); dropping infoboxes, navboxes, tables, reference lists
and footnote markers, `See also`/`External links`/`Further reading`/`Notes` sections,
image captions, edit links, math markup.

Wiki-links are kept as **character offsets into the cleaned prose**, not inline markup,
so every downstream stage treats the text as plain prose while still recovering exactly
which span links to which article.

**Measured:** 15,158 documents, 47,566 sections, 443,527 wiki links, 53.1M characters
(~10.6M words). Artifact scan across the full corpus found 0 residual boilerplate
markers (`[edit]`, `Archived from`, `^ a b`, `.mw-parser`); the 105 `ISBN` / 40 `doi:`
hits inspected were legitimate prose, not reference-list leakage.

**Fixed (2026-09-28):** `bs4.Comment` subclasses `NavigableString`, so the cleaner's
text-walk could not tell an HTML comment from real text — editors' "deleted image
removed" notes and disabled draft sections (raw, unrendered wikitext) were being
appended into the prose. 1,461 of 15,158 articles (9.6%) carried comment nodes, ~317K
characters leaking in. Comments are now stripped before extraction; re-running `clean`
reproduced the identical document/section/link counts above, confirming no real content
was lost. See spec 0001 Changelog.

### 2.4 Licensing

Wikipedia text is CC BY-SA 4.0 — attribution and share-alike apply to any derived data
published from this pipeline (paper figures, released datasets).

---

## 3. Entity & mention layer (spec [0002](specs/0002-entity-mention-layer.md))

### 3.1 Node scope

Graph nodes are **any human mentioned**, not just the 15,158 seed physicists — only
6.1% of link instances point at a seed physicist, so a seed-only graph would discard
most of the social network (collaborators outside physics, spouses, students).
`entities.jsonl` carries an `in_seed` flag so a physicist-only view is still available as
a secondary analysis.

### 3.2 Link resolution → Wikidata

Every distinct wiki-link target (125,927 of them) is resolved title → QID → `P31 = Q5`
in two passes: (1) `action=query&prop=pageprops` in batches of 50 titles against the
Wikipedia Action API; (2) chunked SPARQL `VALUES` queries against WDQS (~2,000 QIDs/
request, POSTed — a GET URL can't hold that many), split into a cheap human-only filter
and a metadata query run only over the humans found.

Chosen over a neural entity linker (BLINK, ReFinED): those solve linking *arbitrary*
text to *all* of Wikidata, a harder problem than needed here, since the mentions are
already human-annotated links — the only open question is "is the target a person."
Cross-checked against the alternative `wbgetentities` API on 1,500 QIDs: 1,499/1,500
agreed, and all 3 disagreements were reader bugs in the discarded method
(deprecated-rank claims, non-preferred-rank dates, Julian/Gregorian date normalization) —
documented in the spec as the reason SPARQL's `wdt:` prefix (which applies Wikidata's
rank and calendar rules) was kept as the resolution path.

**Measured:** 13.3% of link instances (58,989 of 443,527) resolve to a human, across
25,826 distinct people. 4.5% of link targets (5,636 of 125,927) are red links (no
article yet) — real people, currently dropped as a quantified recall loss.

### 3.3 Sentence segmentation

**spaCy `en_core_web_sm`, `senter` pipe only** (no NER, no parser — a 12 MB model,
CPU-only). Run **per paragraph, not per section**: running it on whole sections let the
model run straight through blank lines and weld the last sentence of one paragraph to
the first of the next (measured before the fix: 9.1% of sentences, 31,744 of 349,708,
contained a newline). Paragraph breaks are applied as boundaries directly rather than
left to the model.

**Measured:** 384,656 sentences, 0 containing a newline after the fix.

**Fixed (2026-09-28):** `senter` predicts a sentence boundary right at an open `[` more
often than the text warrants — a bracketed middle initial inside a name ("Mervyn [M.]
Dymally"), or a quoted excerpt with an editorial `[...]` insertion — truncating the
sentence mid-word. Measured before the fix: 1,681 of 384,656 sentences (0.44%) ended on
an unclosed `[`; 135 of 76,205 candidates (0.18%) carried a truncated sentence. Fixed by
merging consecutive senter-predicted sentences while a `[` stays unclosed, capped at 4
merges so a genuinely unclosed `[` typo in a source article can't consume the rest of
its paragraph. Re-run on the full corpus (combined with the clean-stage comment fix
above): 382,404 sentences, 76,216 candidates, **0** with a truncated sentence. See spec
0002 Changelog.

### 3.4 Candidate pair generation

Two rules, scoped to co-occurrence within one sentence:

1. **`subject_link`** — the article's own subject (rarely self-linked in its own
   biography) paired with each linked person in a sentence. The subject QID comes from
   `seed.jsonl`; until coreference resolution exists this is a heuristic without a span,
   tagged explicitly as `mention_type: subject` so its error contribution can be
   measured separately. **Uncapped.**
2. **`link_link`** — two linked people co-occurring in a sentence. **Capped**: above
   `MAX_PEOPLE_FOR_LINK_LINK = 6` distinct people in one sentence, pairs are suppressed
   (counted separately as `link_link_suppressed`) rather than emitted, because
   enumeration sentences ("the seminar was attended by Bohr, Dirac, Yukawa, …") were
   found to produce `C(n,2)` pairs asserting nothing between the listed people —
   measured before capping: 61% of `link_link` pairs (18,524 of 30,612) came from
   sentences listing 4+ people, one sentence alone producing 325 pairs.

Every candidate carries `people_in_sentence`, so the cap can be revisited as a weight
rather than a hard threshold once a gold set exists.

**Measured:** 74,149 mentions (58,991 link + 15,158 subject) over 32,546 distinct people
(15,158 in seed, 17,388 not). **76,205 candidate person-pairs**: 58,713 `subject_link` +
17,492 `link_link` (+ 10,836 suppressed). 31.9% of documents (4,835 of 15,158) yield no
candidate pairs at all — the recall gap inherent to a link-only, no-coreference layer.

### 3.5 Integrity guard

`mentions` compares Wikidata cache coverage against the corpus's link targets before
running and hard-fails (or warns, with `--lenient`) on an incomplete cache — motivated by
an actual incident where a stale 200-row test cache would have silently produced a
near-empty graph rather than an error.

---

## 4. Relation clustering (spec [0003](specs/0003-relation-typology.md), **Superseded**)

**Superseded by §5 below** — kept in full as the historical record of why: even after
entity masking (§4.3), clusters kept cohering around a shared *non-person* entity (a
place, an institution, a topic) rather than a shared relation type. See spec 0003's
Changelog and [finding 0002](findings/0002-entity-masking-effect-on-relation-clustering.md).
Code and the `inpnet cluster`/`cluster-sweep`/`cluster-summary` commands are kept
runnable for reruns/ablation, not removed.

### 4.1 Framing

The layer between candidate pairs and a relation *label* is deliberately
**field-agnostic**: candidate sentences are grouped by unsupervised clustering with no
relation labels presupposed, and a documented plug-in contract (§4.4) is what a
domain-specific typology (physics-specific labels — a follow-on, not-yet-written spec)
must satisfy to consume the clusters. Rationale to state in the paper: physicist
biographies skew collaboration/mentorship and rarely encode real antagonism, so a
typology fit to this domain would misfit fields where dispute is central — keeping
clustering domain-agnostic keeps that door open.

### 4.2 Unit of clustering

Candidate pairs collapse to **distinct sentences** before clustering (76,205 pairs →
43,664 unique sentences — a sentence naming 3+ people yields several pairs). Clustering
runs once per sentence; every candidate pair inherits its sentence's `cluster_id` by
join. Known limitation carried forward explicitly: a sentence asserting two *different*
relations about two different pairs ("X married Y; both later collaborated with Z")
gives X–Y, Y–Z and X–Z the same cluster.

### 4.3 Method: embed → reduce → cluster (BERTopic pattern)

| Step | Technology | Notes |
|---|---|---|
| Embedding | `sentence-transformers`, model `all-MiniLM-L6-v2` (~80 MB, CPU-only) | First ML dependency in the project. Chosen so semantically equivalent phrasing ("his wife" / "married") lands close in embedding space — a lexical method (TF-IDF) would not. |
| Dimensionality reduction | UMAP | `metric="cosine"` — matches how sentence-transformer embeddings are trained/compared; UMAP's Euclidean default was tried first and collapsed the corpus into one cluster (see §4.5). |
| Clustering | HDBSCAN, via `sklearn.cluster.HDBSCAN` (not the standalone `hdbscan` package — ships a prebuilt wheel) | Density-based so no cluster count `K` is committed upfront; produces a native noise label (`-1`) for sentences that fit no dense group, which is treated as a reportable number, not absorbed into the nearest cluster. |

**Entity masking (spec 0003 §Decision 2a).** Unmasked, sentence embeddings key on *who*
a sentence names as readily as on *what relation* it asserts — Observation 3 below is
exactly this, and the article subject makes it worse since it's named in nearly every
sentence of its own article. Every detected person mention is replaced with a single
generic placeholder (`[PERSON]`) before embedding: linked mentions (`mentions.jsonl`) by
their exact span — converted from section-relative to sentence-local using
`sentences.jsonl` first, since that's how spec 0002 §Decision 4 stores offsets, a
conversion missing from the first implementation (see §4.5) — the article subject (no
span — Wikipedia never self-links) by matching its full name and surname from
`entities.jsonl`. `cluster_summary.jsonl`'s exemplars still show original, unmasked
text; only the embedding input changes. `mentions.jsonl`/`entities.jsonl`/`sentences.jsonl`
are now inputs to the `cluster` stage alongside `candidates.jsonl`; `--no-mask-entities`
disables it for ablation. `relation_typology`'s
layer version moved to `0.0.2` in `data_versions.json` since masking changes the
embeddings; measured effect of the `0.0.2` full-corpus run is in §4.5 below.

Determinism: model version pinned, sentence order fixed (sorted by
`(doc_id, section_idx, sent_idx)`), fixed random seeds for UMAP/HDBSCAN, all recorded in
the run's manifest. `cluster_id` is **run-scoped, not a stable identifier** — a re-run
after any code/data/dependency change requires re-deriving any label mapping built on
top of it.

### 4.4 Hyperparameters — code defaults vs. the actual full-corpus run

The values coded as `DEFAULT_*` in [`src/inpnet/relations/cluster.py`](../src/inpnet/relations/cluster.py)
are **unvalidated proposals**, not what was actually used for the reported run — the
paper should cite the run column, not the code defaults:

| Parameter | Code default | Full-corpus run (cite this) |
|---|---|---|
| `model` | `all-MiniLM-L6-v2` | same |
| `n_neighbors` (UMAP) | 15 | 15 |
| `n_components` (UMAP) | 5 | **15** |
| `metric` (UMAP) | cosine | cosine |
| `min_cluster_size` (HDBSCAN) | 15 | **5** |
| `cluster_selection_method` (HDBSCAN) | `eom` (library default) | **`leaf`** |
| `random_state` | 42 | 42 |

`cluster_selection_method="eom"` (excess of mass — HDBSCAN's own default, favoring the
most *persistent* cluster in the condensed tree) collapsed the corpus into one
43,634-sentence cluster; `"leaf"` (favors the tree's leaves — more, smaller clusters, no
persistence competition against one dominant blob) fixed it. This was diagnosed with a
purpose-built dev tool, `inpnet cluster-sweep`
([`src/inpnet/relations/diagnostics.py`](../src/inpnet/relations/diagnostics.py)): embeds
a sample once, then cheaply sweeps UMAP/HDBSCAN settings against that fixed embedding —
the mechanism worth citing as *how* the run-column hyperparameters were chosen, i.e. by
sweep, not guessed.

**Run command:**
```bash
inpnet cluster --cluster-selection-method leaf --n-neighbors 15 --n-components 15 --min-cluster-size 5
```

### 4.5 Measured results

**`0.0.1`, unmasked:** 1,286 clusters, 27,881 noise sentences (**63.8%** of 43,664),
median cluster size 9, 609 clusters ≥10 sentences, full run in 4m42s. Spot-checked
exemplars read as genuinely relation-like: a doctoral-mentorship cluster ("his doctoral
students include…"), a postdoc-supervision cluster, a family-relation cluster ("grandson
of," "brother of"), and named research-collaboration clusters (DNA structure discovery;
black-hole physics) — alongside some clusters that read as shared historical/topical
narrative rather than one specific relation, and clusters that track a shared named
entity rather than a shared relation type (see
[finding 0001](findings/0001-relation-clustering-entity-and-template-bias.md)) — the
motivation for entity masking, §4.3 above.

**`0.0.2`, same hyperparameters, entity masking on:** 1,232 clusters, 28,542 noise
(**65.4%**); 43,648 of 43,664 sentences (99.96%) had at least one person mention
actually masked. (An earlier version of this run had a bug — a link mention's
section-relative span, spec 0002 §Decision 4, was sliced directly against sentence-local
text, so most linked-person names were silently left unmasked; fixed by converting
through each sentence's own start in `sentences.jsonl`, now a fourth input to `cluster`.
See spec 0003 Changelog and
[finding 0002](findings/0002-entity-masking-effect-on-relation-clustering.md)'s
addendum.) Checked directly against two concrete examples that motivated and then
exposed the bug — the three Henry B. Eyring sentences in spec 0003 §Decision 2a, and
three "Sir Humphry Davy" sentences from unrelated articles: **all six are now noise
(`cluster_id -1`) in the corrected run, no longer clustered with each other or with
anything else.**

**Not yet done, and load-bearing for the paper's validity claims, for both versions:**
the two-annotator human coherence review (spec 0003 §Decision 5) and the determinism /
byte-identical rerun check are still open acceptance criteria. Report cluster numbers as
provisional until that review lands.

### 4.6 The typology plug-in contract

A domain-specific relation typology (e.g. the physics label set, not yet written) is one
YAML config per `docs/specs/typologies/<name>.yaml`
([schema template](specs/typologies/TEMPLATE.yaml)): each relation type names its
`cluster_ids` from one named `clustering_run_id`, plus optional Wikidata `P`-ids for
distant supervision and later novelty measurement (how much of the extracted graph has
no Wikidata equivalent). A validator rejects a config referencing a `cluster_id` absent
from its named run. This file is meant to be the single source the annotation guide, any
model prompt, and the evaluation script all read — not hand-copied into three places.

---

## 5. Relation attribute detection & assembly (specs [0004](specs/0004-relation-attribute-detection.md), [0005](specs/0005-schemaless-relation-attributes.md), Accepted)

### 5.1 Framing

Collapsing everything a sentence asserts into one embedding is what made §4's clusters
key on incidental shared context (a place, an institution) instead of relation type.
This layer replaces that with two stages: pull a sentence's individual factors —
when, where, at what institution, doing what — out as separately detected, typed spans
(`detect-attributes`, spec 0004), then assemble participants + spans into one relation
record per candidate pair, an open bag of attributes rather than a fixed triple
(`assemble-relations`, spec 0005). "Schemaless" means the *set* of attribute types is
open and can vary by domain — the record container is fixed, what fills it isn't.

Candidate sentence/pair finding is unchanged from §3.4 (wikilink-derived) — real prose
rarely links enough of a relation's surrounding factors in one sentence for links to
serve as a general detector of them, so links remain for finding candidate sentences,
not for typing what's inside them.

### 5.2 Detection: spaCy NER + dependency parse

Same pinned model as segmentation (`en_core_web_sm`), with `parser`+`ner` enabled instead
of `senter` alone — real throughput cost §3.3 never had to pay, since sentences are
processed per-sentence, not per-paragraph, here.

| `attr_type` | Mechanism |
|---|---|
| `time` | NER label `DATE` |
| `place` | NER labels `GPE`/`LOC`/`FAC` |
| `institution` | NER label `ORG` |
| `action` | Shortest dependency-parse path between the two participant tokens; the minimal span enclosing that path's leftmost-to-rightmost token |

`time`/`place`/`institution` are **sentence-scoped** (true regardless of which candidate
pair asks); `action` is **pair-scoped** (depends on which two participants), since one
sentence can name several pairs with different actions between them —
`attribute_spans.jsonl` carries `candidate_id` (null for sentence-scoped rows) to keep
the two apart. `topic` was raised during design (it's what drove several of §4's
non-person-entity clusters) and deliberately dropped from this first pass — no reliable
text-only signal identified yet.

### 5.3 Assembly and the coverage split

One relation record per candidate pair: its two participants (located the same way spec
0003 §2a's masking located them — a linked participant's span, converted section-relative
→ sentence-local; the article subject by name match, since it carries no span) plus
whatever attributes spec 0004 found in its sentence.

`coverage` = fraction of the sentence's **content tokens** (non-stopword, non-punct) that
fall inside a participant or attribute span — token-based rather than character-based, so
a long institution name doesn't dominate the number regardless of relevance. Records
below `--min-coverage` (default `0.5`, a proposal to tune, not a measured result — same
posture §4.4's hyperparameters took) go to `low_coverage_relations.jsonl` instead of
`relations.jsonl`, each with a human-readable text rendering
(`relations_summary.txt`/`low_coverage_relations_summary.txt`) showing the sentence with
every attribute inline *and* whatever content wasn't captured, spelled out rather than
left implicit in the coverage number.

### 5.4 Measured results

**`0.0.1` (full corpus):** 43,664 unique sentences, 76,205 candidate pairs → 131,419
attribute spans (institution 41,541, time 28,702, place 18,743, action 42,433). `action`
found for only 55.7% of pairs. A full diagnostic over all 33,758 misses (dev script, not
a pipeline stage) found this **wasn't** mostly non-verbal relations as first guessed:
**95.6% (42.4% of every candidate pair in the corpus) traced to one cause** — a
`subject_link` candidate whose subject is referred to only by pronoun in that sentence
("He did doctoral research under..."), never by name. This is spec 0002's
already-documented coreference gap (§Non-goals), just never quantified before. Only 4.3%
of misses (1.9% of all candidates) had both participants resolved with no dependency
path found — genuinely mostly appositive/non-verbal family relations as originally
expected, plus one small real bug (the senter not splitting on `"Charles I."`-style
abbreviations) and a handful of sentences truncated mid-wikimarkup (spec 0001/0002
cleaning edge cases, ~0.1% of candidates, not yet fixed).

**`0.0.2` (full corpus, adds pronoun-resolution fallback):** assumes the earliest
third-person pronoun in a sentence refers to the article subject when name-matching
fails (gender-aware via `entities.jsonl`'s `gender` field where known) —
**an accepted-for-now heuristic, explicitly flagged as a pitfall, not a validated fix**:
a sentence can pronoun-reference someone other than the subject, and this can't tell.
Every affected `attribute_spans.jsonl` row, manifest count, and assembled relation
record carries `pronoun_resolved: true` rather than looking identical to a verified
match. Result: `action` found for **85.9%** of candidate pairs (65,443 of 76,205, up
from 55.7%), of which **35.2%** (23,010) rest on the heuristic. `action` spans still
average ~94 characters against a ~187-character sentence — the enclosing-span design
(spec 0004 §Decision) is unchanged, still broader than a tight verb phrase, still a
measured characteristic rather than a bug. Layer version bumped `0.0.1` → `0.0.2` (a
real methodological decision, not a bug fix); `0.0.1`'s output is kept for comparison.

**Assembly** (`assemble-relations`, full corpus, `--min-coverage 0.5`): `0.0.1` gave
76,205 candidate pairs → 37,829 relations, 38,376 low-coverage (50.4%), mean coverage
0.510. Reading a sample of `low_coverage_relations_summary.txt` confirmed the split
does its intended job — low-coverage records mostly showed a correctly-detected
`institution`/`time` alongside a *missing* `action`, not short or garbled sentences.
**`0.0.2`**, after the pronoun fix: **51,148 relations**, **25,057 low-coverage
(32.9%)**, mean coverage **0.624**; 30.3% of all records (23,091 of 76,205) carry
`pronoun_resolved: true`. Determinism reconfirmed end to end (0004 → 0005 rerun,
byte-identical, same as `0.0.1`).

**Not yet done:** the two-annotator-style qualitative review this layer's design
implies (spot-checks above are informal, single-reviewer); a systematic count of how
often `action`'s broad span overlaps attributes already typed elsewhere in the same
record; and a measured *error* rate for the pronoun heuristic itself (only its *usage*
rate — 30.3%/35.2% above — is currently tracked, not how often it's actually wrong).

---

## 6. Software & environment

| | |
|---|---|
| Language / runtime | Python 3.13 |
| Environment / dependency manager | [`uv`](https://docs.astral.sh/uv/) (`uv sync`, `uv run inpnet ...`) |
| Hardware at time of writing | 16 CPU cores, **no GPU** — every model choice through spec 0003 is explicitly CPU-feasible for that reason |
| CLI entry point | `inpnet` (`src/inpnet/cli.py`) |

**Core dependencies** (`uv sync`): `requests` (HTTP), `beautifulsoup4` + `lxml` (HTML
cleaning), `spacy` + `en_core_web_sm` (sentence segmentation — spec 0002 — and, with more
of the same pinned model's pipeline enabled, attribute detection — spec 0004), `pyyaml`
(typology config, spec 0003). Specs 0004/0005 needed no new dependency: `parser`/`ner`
were already part of `en_core_web_sm`, just excluded by spec 0002's `senter`-only load.

**Optional `relations` extra** (`uv sync --extra relations`, needed only for the
`cluster` stage): `sentence-transformers` (sentence embeddings), `torch` (CPU-only build,
pinned via a scoped `uv` index to avoid the CUDA wheel), `umap-learn`, `scikit-learn`
(`sklearn.cluster.HDBSCAN`), `numpy`. Kept optional because it is the project's first ML
dependency, pulls in a few hundred MB, and is not needed by anyone working only on
corpus acquisition or the mention layer.

Exact version floors: see [`pyproject.toml`](../pyproject.toml).

---

## 7. Reproducibility & provenance

- **Revision pinning**: every fetched article records its Wikipedia revision id (from
  the REST API's `ETag`) and a permanent `oldid` URL — the corpus is a citable snapshot,
  not "whatever Wikipedia says today."
- **Wikidata snapshot**: `wikidata_cache.jsonl.gz` (3.7 MB) is tracked in git, so
  `segment`/`mentions` reruns need no network access and reproduce against the exact
  Wikidata state used for the paper.
- **Manifests**: every stage writes `_manifest.json` — input hashes, config, tool/model
  versions, timestamps, output counts.
- **Determinism**: fixed seeds throughout (spaCy is deterministic by construction;
  UMAP/HDBSCAN seeded via `random_state`); byte-identical reruns are a stated acceptance
  criterion for the corpus and mention stages (verified), specs 0004/0005 (**verified**
  — full-corpus rerun of `detect-attributes` and `assemble-relations` produced SHA-256-
  identical `attribute_spans.jsonl`, `relations.jsonl` and `low_coverage_relations.jsonl`),
  and an open one for spec 0003's superseded clustering (never verified before it was
  superseded — §4).
- **Hand-labelled data** (`data/annotations/`) is the one directory under `data/` tracked
  in git — everything else is gitignored and re-derivable from raw sources plus the
  pipeline.

---

## 8. Known limitations to state in the paper

- **Selection bias**: Wikidata's `P106` occupation property under-covers people English
  Wikipedia's category tree would call physicists; the gap is unmeasured (open question
  in spec 0001).
- **Demographic/coverage bias**: Wikipedia's physicist coverage skews male, Western, and
  modern (era). `entities.jsonl` carries gender/birth-year for a quantified bias
  analysis; that analysis itself is not yet run.
- **Recall ceiling of the link-only mention layer**: ~3.8 person-links per article is a
  precise floor, not a complete extraction — most relation partners appear as pronouns or
  unlinked names, invisible to this layer. A GPU-based NER/coreference layer is the
  documented (not yet built) way to raise recall.
- **Subject heuristic over-firing**: without coreference, the article-subject pairing
  rule fires on every sentence with a person-link, including sentences not actually about
  the subject; every candidate is tagged with its rule so this can be measured and
  ablated later.
- **Red-linked people** (5,636 targets, 5,998 mentions) are real, named people dropped
  for lack of a Wikidata item — a quantified, not hypothetical, recall loss.
- **Cluster-based relation typing (spec 0003) is superseded**, not merely unverified —
  see §5.1 and spec 0003's Changelog for why (clusters kept cohering on a shared
  non-person entity rather than a shared relation type, even after person-masking).
  [`docs/findings/0001-relation-clustering-entity-and-template-bias.md`](findings/0001-relation-clustering-entity-and-template-bias.md)
  and [`0002`](findings/0002-entity-masking-effect-on-relation-clustering.md) document
  the evidence that motivated replacing it. Its code is kept for reruns/ablation.
- **`action` attribute recall is 85.9%** as of `0.0.2` (up from 55.7% in `0.0.1` — see
  §5.4), but **35.2% of that recall rests on an explicitly unvalidated heuristic**: a
  pronoun in the sentence is assumed to refer to the article subject whenever name
  matching fails. This isn't a detection improvement in the normal sense — it's spec
  0002's coreference gap, newly quantified (42.4% of *every* candidate pair) and worked
  around with an assumption known to be sometimes wrong. Every affected row/record is
  flagged (`pronoun_resolved: true`), but how often the assumption is actually *wrong*
  is not yet measured — only how often it's used. Real coreference resolution remains
  the single highest-leverage open item for this layer.
- **`action` spans are broad**, averaging about half the sentence's length (~94 of ~187
  characters) — the shortest-dependency-path span often reads as a full clause rather
  than a tight verb phrase, and can overlap `time`/`place`/`institution` spans already
  typed elsewhere in the same record; not yet separately quantified (§5.4).
- **Two small spec 0001/0002 bugs**, found via the `action`-miss diagnostic rather than
  their own testing: sentences occasionally truncated mid-wikimarkup at a stray `"["`
  (~0.1% of candidates), and the senter not splitting on `"Charles I."`-style
  capital-letter-plus-roman-numeral abbreviations, merging two real sentences into one.
  Both low-volume, neither fixed yet.
- **`topic` is not detected** (spec 0004 Non-goals) even though it was part of what
  motivated moving off whole-sentence clustering in the first place (shared topics like
  "crystal growth" were as much a source of spurious cohesion as shared places) — no
  reliable text-only signal was identified for a first pass; open for a follow-on.
- **Relations are model output, not fact.** Any published graph or figure must label
  extracted edges as such (AGENTS.md §6).

---

## 9. Status snapshot

| Stage | Spec | Spec status | Implemented & run on full corpus | Key open item |
|---|---|---|---|---|
| `seed`/`fetch`/`clean` | [0001](specs/0001-corpus-acquisition.md) | Draft | Yes | Redirect handling incomplete in `fetch_log.jsonl`; strict-vs-subclass `P106` scope needs a human decision |
| `resolve`/`segment`/`mentions` | [0002](specs/0002-entity-mention-layer.md) | Draft | Yes | Red-link handling, sentence-vs-paragraph pairing, `MAX_PEOPLE_FOR_LINK_LINK` tuning all open |
| `cluster` | [0003](specs/0003-relation-typology.md) | **Superseded** (0004, 0005) | Yes (1,286 clusters, 63.8% noise) | Superseded, not being carried forward — kept for reruns/ablation only |
| `detect-attributes` | [0004](specs/0004-relation-attribute-detection.md) | **Accepted** | Yes, `0.0.2` (154,429 spans; action found for 85.9% of pairs, 35.2% via the pronoun heuristic) | Real coreference resolution (42.4% of pairs gated on it); `topic` detection unstarted |
| `assemble-relations` | [0005](specs/0005-schemaless-relation-attributes.md) | **Accepted** | Yes, `0.0.2` (51,148 relations, 32.9% low-coverage) | `--min-coverage` untuned past its default; low-coverage review is informal so far |
| Physics relation typology | *(not yet written)* | — | No | Follow-on spec; would now plug into `assemble-relations`'s `action` attribute, not whole-sentence clusters |

---

## References

Full citations and alternatives-considered tables live in the specs themselves — this
digest intentionally omits them to stay short. See:
[0001](specs/0001-corpus-acquisition.md#references),
[0002](specs/0002-entity-mention-layer.md#references),
[0003](specs/0003-relation-typology.md#references) (superseded),
[0004](specs/0004-relation-attribute-detection.md#references),
[0005](specs/0005-schemaless-relation-attributes.md#references).
