# 0002 — Entity & mention layer

| | |
|---|---|
| **Status** | Draft |
| **Depends on** | 0001 |
| **Superseded by** | — |
| **Owner** | Falk Minks |
| **Last updated** | 2026-08-12 |

## Context

Spec 0001 produced 15,158 cleaned articles — 47,566 sections, 443,527 wiki links,
53.1 M characters of prose. Nothing downstream can start until we know **which spans
refer to which people**: relation extraction is a function of entity pairs, and the
current zero-shot RE models make that ordering explicit (GLiREL takes pre-identified
head/tail entities as input).

Wikipedia gives this project an advantage a general-domain pipeline does not have. Those
443,527 links are **human-curated entity links**: a link to `./Niels_Bohr` is an entity
annotation a person wrote deliberately. Measured on a 100-instance random sample,
**13% of link instances point at a human** (`P31 = Q5`), projecting to roughly
**57,700 person-links** corpus-wide, about 3.8 per article — carrying *zero model error*.

No GPU is available at present (16 CPU cores); one is expected later. This spec therefore
harvests only the model-free layer, and fixes a schema that later model-derived mentions
can be appended to without reshaping the file.

## Goal

Identify person mentions with Wikidata QIDs and the candidate person-pairs they imply,
using no statistical model — fast, deterministic, and free of NER error.

## Non-goals

- **Coreference and NER.** Deferred to the GPU stage. The schema reserves their enum
  values so adding them later is an append, not a migration.
- Relation extraction, and the relation typology — separate specs.
- Cross-document entity merging beyond QID identity. Two mentions sharing a QID are the
  same person; no cleverer merging is attempted.
- Linking mentions to people who have no Wikidata item.

## Decision

### 1. Node scope: any human mentioned

Nodes are **not** restricted to the 15,158 seed physicists. Only 6.1% of link instances
point at a seed physicist, so a seed-only graph would discard most of the social network —
collaborators in other fields, spouses, students, rivals. Non-seed humans become nodes
described only as other articles describe them; `entities.jsonl` carries an `in_seed`
flag so a physicist-only view remains available for a secondary analysis.

### 2. Resolve link targets through Wikidata

125,927 distinct link targets are resolved title → QID → `P31 = Q5`, in two passes:

**Pass 1 — title → QID.** `action=query&prop=pageprops&ppprop=wikibase_item` on
en.wikipedia.org, 50 titles per request (2,519 requests).

**Pass 2 — QID → claims.** Chunked SPARQL `VALUES` against WDQS, ~2,000 QIDs per
request. Two queries, not one, because filtering first shrinks the expensive one:

```sparql
-- A: which are human?  Cheap: no OPTIONALs, no property paths.
SELECT ?item WHERE { VALUES ?item { wd:Q1 ... } ?item wdt:P31 wd:Q5 }

-- B: metadata, for the humans only (~13% of the input)
SELECT ?item ?birthYear ?deathYear ?gender ?occ WHERE { VALUES ?item { ... }
  OPTIONAL { ?item wdt:P569 ?b . BIND(YEAR(?b) AS ?birthYear) } ... }
```

Neither uses `GROUP_CONCAT` nor `SERVICE wikibase:label` — both are what pushed the spec
0001 seed query past the WDQS timeout. Queries beyond 2 KB are **POSTed**, since a
`VALUES` clause of thousands of QIDs exceeds practical GET URL limits.

**Timeout handling.** `sparql()` raises `QueryTruncated` on WDQS's HTTP-200 truncation;
each chunk that trips it is **halved and retried** down to a floor of 25, so a slow
region degrades gracefully instead of failing the run. (Measured: zero halvings needed
at 2,000 per chunk.)

**Three states, not a boolean.** A title is `claims_resolved` if it has no Wikidata item
at all (definitively unidentifiable) *or* its claims were fetched. Only "has a QID whose
claims were never fetched" is unresolved, and `is_human` is then `null` rather than
`false`. Collapsing those is what let an incomplete cache silently drop real people.

Results cache to `title_qid.jsonl` and `qid_claims.jsonl` — both resumable — and join
into `wikidata_cache.jsonl`. Paid once: a re-run with both caches warm completes in
**5 seconds** with no network.

Using the curated links rather than a neural entity linker (BLINK, ReFinED) is a
deliberate trade: those models exist to link *arbitrary* text to *all* of Wikidata, while
here the mentions are already annotated and the only open question is whether the target
is a person. Dictionary-grade resolution answers that exactly, and faster.

### 3. The article subject is the implicit second party

A biography's subject is mentioned constantly — and almost never linked in its own
article, because Wikipedia does not link a page to itself. It is also the party in most
relations the article asserts ("He worked with X", "She succeeded Y").

So each document contributes its own subject QID (known from `seed.jsonl`) as a
document-level entity. Until coreference lands, the subject has no span; it pairs with
linked people at sentence granularity. This is the **subject heuristic**, and it is
recorded as its own `mention_type` so its contribution can be ablated and measured
separately rather than baked in invisibly.

### 4. Sentence segmentation

Candidate pairs are scoped to sentences, so boundaries are needed. spaCy `en_core_web_sm`
with only the `senter` pipe enabled. A 12 MB model, no GPU, and it handles the
abbreviation cases ("Dr.", "et al.", "Ph.D.") that a regex splitter gets wrong.

**Segmentation runs per paragraph, not per section.** Feeding whole sections in — with
paragraphs joined by `\n\n` — let the senter run straight through the blank line: **9.1%
of sentences (31,744 of 349,708) came out containing a newline**, welding the last
sentence of one paragraph to the first of the next. That inflates the co-occurrence
window and manufactures pairs between people who never share a sentence. A paragraph
break is an unambiguous boundary, so it is applied directly rather than left to the model.

Offsets remain **relative to section text** — the paragraph base offset is added back —
so the link offsets from spec 0001 stay valid. `sent_idx` runs across the whole section
rather than restarting per paragraph.

### 5. Candidate pairs

- **subject <-> linked person** in the same sentence — the dominant case in biography
- **linked person <-> linked person** in the same sentence

Counts are reported per rule; that number sizes the entire extraction stage.

**Enumerations are capped.** A manual read of `link_link` pairs found the rule's dominant
failure mode: sentences that *list* people. "The most prominent physicists participated in
the seminar: Niels Bohr, Paul Dirac, Hideki Yukawa, Julian Schwinger, Abdus Salam, Aage
Bohr, Ilya Prigogine..." yields C(n,2) pairs, and essentially none of them assert a
relation *between* the listed people — each relates to the article's subject.

Measured before capping: **61% of `link_link` pairs (18,524 of 30,612) came from sentences
listing four or more people**, and a single sentence produced **325 pairs**.

So above `MAX_PEOPLE_FOR_LINK_LINK` (6) distinct people in a sentence, `link_link` pairs
are suppressed and counted in `link_link_suppressed`. `subject_link` is deliberately
**not** capped — in exactly those sentences, subject-to-each is the relation the text
does assert. Every candidate also carries `people_in_sentence`, so a later stage can
weight or filter on it rather than trusting the threshold blindly.

### 6. `mentions` hard-fails on an incomplete cache

A short cache is indistinguishable from a corpus full of non-people: unresolved titles
produce no mentions, the run succeeds, and the graph comes out near-empty. `mentions`
therefore compares the cache against the corpus's link targets before doing any work and
exits with a count and examples if coverage is short. `--lenient` downgrades it to a
warning for deliberate subset runs.

This is not hypothetical: a 200-row cache left over from a `resolve --limit 200` test was
sitting on disk, and running `mentions` against it would have produced a plausible-looking
but almost empty network.

### 7. Provenance and archiving

Spec 0001 pins every article to a revision id; Wikidata is equally mutable and needs the
same treatment. Every cache record carries `fetched_at`, and the manifest records start
and finish timestamps plus both chunk sizes.

`wikidata_cache.jsonl.gz` (3.7 MB) is written beside the plain file and **tracked in
git**. It is the citable snapshot, and it lets anyone re-run `segment` and `mentions`
with no network access at all. (Lives at
`data/interim/entity_mention_layer/{version}/wikidata_cache.jsonl.gz`; `.gitignore`
un-ignores that filename under any version directory.)

## Interface

Three stages, separate commands, each writing `_manifest.json` — same pattern as 0001.

| Stage | Reads | Writes |
|---|---|---|
| `resolve` | `documents.jsonl` (spec 0001's layer) | `wikidata_cache.jsonl` |
| `segment` | `documents.jsonl` (spec 0001's layer) | `sentences.jsonl` |
| `mentions` | above + `seed.jsonl` | `mentions.jsonl`, `entities.jsonl`, `candidates.jsonl` |

All three stages write under this layer's own version directory,
`data/interim/entity_mention_layer/{version}/`, and read `documents.jsonl` from
`data/interim/corpus_acquisition/{version}/` — each `{version}` taken from its layer's
entry in [`data_versions.json`](../../data_versions.json), bumped by hand. See AGENTS.md §4.

**`mentions.jsonl`** — enum values marked with `*` are unused now and filled by the GPU
stage later:

```jsonc
{
  "mention_id": "string — '<doc_id>:<section_idx>:<start>'",
  "doc_id": "string",
  "section_idx": "int",
  "sent_idx": "int — sentence index within the section",
  "start": "int|null — char offset into section text; null for subject mentions",
  "end": "int|null",
  "surface": "string",
  "entity_id": "string — QID",
  "mention_type": "enum — link | subject | ner* | coref_pronoun* | coref_nominal*",
  "link_method": "enum — wikilink | subject | alias* | model*",
  "confidence": "float — 1.0 for wikilink"
}
```

**`entities.jsonl`** — one per distinct human: `qid`, `canonical_name`, `in_seed`,
`mention_count`, `doc_ids`, and `birth_year` / `death_year` / `gender` / `occupations`.
Gender and era feed the coverage-bias analysis spec 0001 flagged as mandatory.

**`candidates.jsonl`** — `sentence`, `head_entity_id`, `tail_entity_id`, spans where
known, `doc_id`, `sent_idx`, and `rule` (`subject_link` or `link_link`).

## Alternatives considered

| Option | Why not (for now) |
|--------|-------------------|
| `wbgetentities` for pass 2 | 50 QIDs per request — 1,954 requests versus ~50 for SPARQL. It was implemented first and then **cross-checked against SPARQL on 1,500 QIDs: 1,499 agreed, and all three discrepancies were cases where SPARQL was right.** See §Cross-check below. |
| Neural entity linker (BLINK, ReFinED) over all Wikidata | Solves a harder problem than we have — the mentions are already annotated by humans; only "is this a person?" is open. |
| NER + coreference first | Needs a GPU we do not have yet, and would mix model error into the layer everything else is built on. Links give a clean floor to measure the models against later. |
| Restrict nodes to seed physicists | Discards ~94% of person-links. See §Decision 1. |
| Regex sentence splitting | Fails on "Dr.", "et al.", "Ph.D." — frequent in this corpus. |
| Pair within whole sections instead of sentences | Sections run to thousands of characters; co-occurrence at that distance is not evidence of a relation. |

## Cross-check: SPARQL vs wbgetentities

Before discarding the `wbgetentities` results, 1,500 QIDs resolved by both methods were
compared. **1,499 of 1,500 agreed on `is_human`.** Every discrepancy favoured SPARQL,
and each exposed a real bug in the claim parser:

| QID | What it is | Cause | Correct |
|---|---|---|---|
| Q17021508 | Petróczy-Kármán-Žurovec — a **helicopter** | Carries a *deprecated* `P31 = Q5`; the parser read every claim regardless of rank and called it a person | SPARQL |
| Q178217 | Diophantus of Alexandria | Two birth dates, normal `+0201` and **preferred** `+0200`; `wdt:` takes preferred, the parser took list order | SPARQL |
| Q1232515 | Dmitry Blokhintsev | Born 29 Dec 1907 in the **Julian** calendar (`Q1985786`) = 11 Jan 1908 Gregorian; WDQS normalises, raw claims do not | SPARQL |

SPARQL's `wdt:` prefix applies Wikidata's rank rules and calendar conversion natively,
which raw claim reading has to reimplement. `_claim_ids` and `_claim_year` were fixed to
respect rank, with regression tests for the helicopter and Diophantus cases. The
Julian/Gregorian difference is inherent and documented rather than "fixed" — a
year-boundary shift is immaterial to era-level bias analysis.

Because the old results carried the rank bug, all 27,691 of them were **discarded and
re-resolved** through SPARQL rather than left in a mixed-method cache.

## Red links

4.5% of link targets (5,636 of 125,927, 5,998 link instances) are **red links** —
Wikipedia's "create this page" URLs for people with no article yet. Their hrefs look
like `./Wolfgang_Riezler?action=edit&redlink=1`, and the cleaner originally kept the
query string, producing unusable pseudo-titles. `_title_from_href` now strips it.

They account for nearly all of the 5,768 targets with no Wikidata item. None were
misclassified as human, so the graph was never corrupted — but the names are real
physicists (Wolfgang Riezler, Peter Pringsheim) who are genuinely mentioned and
currently dropped. This is a **quantified recall loss**, not an unknown: see *Open
questions*.

## Risks

- **Recall is low by construction.** ~3.8 person-links per article cannot capture what a
  biography asserts; most relation partners appear as pronouns or unlinked names. This
  layer is a precise floor, not a complete one — the GPU stage exists to raise recall, and
  the gap between them is itself a reportable number.
- **The subject heuristic over-fires** in sections about someone else. Without coreference
  it fires per sentence containing a person-link, which will sometimes pair the wrong two
  people. Mitigated by tagging the rule on every candidate so it can be ablated and its
  error rate measured.
- **Link bias**: Wikipedia links a person on first mention and often not again, so
  repeated relation partners are under-counted relative to how often they are discussed.
- **13% is a sample estimate** from 100 instances. The real rate could differ by a few
  points; the resolver measures it exactly, and a large deviation means the resolver is
  wrong rather than the estimate.

## Acceptance criteria

- [x] `wikidata_cache.jsonl` resolves all 125,927 distinct link targets. Measured
      human-link rate **13.3%** (58,989 of 443,527 instances) against the 13% sampled
      estimate, across **25,826** distinct people. `claims_unresolved: 0`.
- [x] `mentions.jsonl`, `entities.jsonl`, `candidates.jsonl` produced for the full corpus:
      **74,149 mentions** (58,991 link + 15,158 subject) over **32,546 people**
      (15,158 in seed, 17,388 not), and **76,205 candidate pairs**.
- [x] Offsets round-trip **and** carry no surrounding whitespace: 58,991 link mentions
      checked, **0 failures**. `links_outside_sentence: 0`.
- [x] Re-running `mentions` on unchanged inputs is byte-identical (SHA-256 over
      `mentions`, `entities` and `candidates`).
- [x] Candidate counts per rule: `subject_link` **58,713**, `link_link` **17,492**,
      `link_link_suppressed` **10,836**. **4,835 of 15,158 documents (31.9%) yield no
      pairs** — the recall gap this layer was expected to have; see Risks.
- [x] Manual read of `link_link` samples. This found the two defects that shaped
      §Decision 4 and §Decision 5 (paragraph-spanning sentences, enumeration blowup) and
      was the highest-value check in the spec. After the fixes the same sample surfaces
      genuine relations — "another doctoral student of Julian Schwinger", "His advisors
      were Max Planck and Heinrich Rubens", "a nephew of physicist Heinrich Hertz".
- [x] Unit tests: **74 passing**, covering QID resolution, the `P31 = Q5` filter with
      rank handling, VALUES chunking and truncation-halving, paragraph segmentation,
      mention construction, pair generation, and the cache-coverage guard.

## Open questions

- **Eponymous mentions.** A person-link is sometimes an award, institute or law *named
  after* someone rather than the person acting: "awarded him the Yuri Gagarin Medal and
  the Vikram Sarabhai Medal" pairs two people who have no relation to each other.
  Measured at **244 of 58,991 person-link mentions (0.4%)** — cues are `Award`, `Prize`,
  `Medal`, `Institute`, `Lecture`, `Laboratory`, `Professorship`. Small enough to leave
  to the relation classifier for now; revisit if it shows up in error analysis.
- **Sentence-level or paragraph-level pairing?** Sentences are conservative; some
  relations span a boundary ("He moved to Copenhagen. There he worked with Bohr.").
  Revisit once coreference exists, since that example is really a coref problem.
- **Is `MAX_PEOPLE_FOR_LINK_LINK = 6` the right cut?** It was chosen from the pair-count
  distribution, not from measured precision. Once a gold set exists, tune it — or replace
  the hard cut with a weight over `people_in_sentence`, which every candidate already carries.
- Should the subject heuristic fire on **every** sentence with a person-link, or only
  those with a subject-referring pronoun? The latter needs coref; the former is what
  ships now, and its precision should be measured before it is trusted.
- **Red-linked people.** 5,998 link instances name people with no article and therefore
  no QID — Wolfgang Riezler, Peter Pringsheim and similar. They are currently dropped.
  The surface text gives a usable name, so they could become graph nodes keyed by name
  instead of QID, at the cost of no cross-document identity and no metadata. Worth
  deciding once the graph exists and the loss can be seen in context.
- Redirects among link targets: `redirects=1` resolves them, but two titles then map to
  one QID. That is correct behaviour; worth confirming it does not distort mention counts.

## References

- GLiREL — Generalist Model for Zero-Shot Relation Extraction (NAACL 2025) —
  https://aclanthology.org/2025.naacl-long.418/
- Maverick: Efficient and Accurate Coreference Resolution (ACL 2024) —
  https://aclanthology.org/2024.acl-long.722.pdf
- fastcoref — https://github.com/shon-otmazgin/fastcoref
- MediaWiki API: Pageprops — https://www.mediawiki.org/wiki/API:Pageprops
- Wikidata API: wbgetentities — https://www.wikidata.org/w/api.php

## Changelog

- 2026-09-28 — **fixed: `senter` mis-splits sentences at an open `[`, truncating
  them.** Surfaced by a spec 0004 diagnostic: candidate
  `Q3760460:3:4:Q3760460:Q1922193`'s sentence ended "...formed by Congressman Mervyn
  [" — the real text is "Mervyn [M.] Dymally", a bracketed middle initial inside a
  wikilink's display text. `senter` predicts a sentence boundary right at the `[`
  (also seen after a quote mark introducing an editorial `[...]` insertion in a
  quotation) more often than the text warrants, silently discarding the rest of the
  sentence into a following fragment that no longer reads as one. This compounded
  with the HTML-comment leak fixed in spec 0001's Changelog — un-rendered
  `[[File:...]]`/`[[Page]]` wikitext syntax in leaked comments produced its own
  false brackets — so a chunk of the affected cases at this layer were actually
  0001's bug wearing this one's symptom. Measured on the current (pre-fix, both
  bugs present) `sentences.jsonl`: **1,681 of 384,656 sentences (0.44%)** end on an
  unclosed `[`; on `candidates.jsonl`: **135 of 76,205 candidates (0.18%)** carry a
  truncated sentence — not the ~31 an initial spot check estimated.
  Fixed by merging consecutive `senter`-predicted sentences while a `[` is left
  unclosed, capped at `MAX_BRACKET_MERGE = 4` merges (99.4% of affected groups
  resolve within that, chosen from the corpus's own merge-chain-length
  distribution) so a genuinely unclosed `[` typo in a source article — found one,
  `Q18927072`: "born in ... on [September 15, 1931" and grew up..." never closes in
  that paragraph — cannot swallow the rest of it.
  Re-ran `clean` (spec 0001's fix) → `segment` → `mentions` end to end on the full
  corpus: **382,404 sentences** (1,452 produced by a merge), **76,216 candidates,
  0 affected** by the truncation symptom (down from 135) — one residual sentence
  (not part of any candidate) still ends unclosed, a five-bracket run in `Q524252`
  one merge past the cap; left as a documented, capped residual rather than raising
  the cap further. Regression tests: `test_bracket_split_is_merged_back_together`,
  `test_bracket_merge_is_capped` in `tests/test_segment.py`.
  **Not yet promoted**: this rerun was written to a scratch directory to measure the
  fix, not into `data/interim/{corpus_acquisition,entity_mention_layer}/` as a new
  `{version}` — that regeneration (and the `data_versions.json` bump) is a
  deliberate follow-up, not done as part of this fix.
- 2026-08-12 — created.
- 2026-08-25 — pass 2 rewritten from `wbgetentities` (~1,954 requests) to chunked SPARQL
  `VALUES` (~60), split into a cheap human filter plus a metadata query over the humans
  only. Cross-checked against 1,500 QIDs already resolved by the old method: **1,499/1,500
  agreement**, and all three discrepancies favoured SPARQL —
  - Q17021508 is a *helicopter* carrying a **deprecated** `P31 = Q5`; reading claims
    without regard to rank called it a person. Fixed in `_best_rank`.
  - Q178217 (Diophantus) has a **preferred**-rank birth year that rank-blind parsing missed.
  - Q1232515 is stored as 29 Dec 1907 **Julian**, which WDQS reports as 1908 Gregorian.
- 2026-08-25 — added the cache-coverage guard, `claims_resolved` tri-state, `fetched_at`
  provenance, and the git-tracked `wikidata_cache.jsonl.gz` (3.7 MB) snapshot.
- 2026-08-26 — two defects found by the required manual read, both now fixed and
  regression-tested:
  - **Sentences spanned paragraph breaks** (31,744 of 349,708, 9.1%). Segmentation now
    runs per paragraph; re-segmented to 384,656 sentences with **0** containing a newline.
  - **Enumerations exploded `link_link`** — 61% of pairs came from sentences listing 4+
    people, one sentence yielding 325. Capped at 6 distinct people, `subject_link` left
    uncapped. `link_link` fell 30,612 → 17,492.
- 2026-08-26 — full corpus built: 74,149 mentions, 32,546 people, 76,205 candidate pairs;
  deterministic across runs; 74 tests passing.
- 2026-09-28 — this layer's outputs moved from the flat `data/interim/` into
  `data/interim/entity_mention_layer/{version}/` (starting at `0.0.1`), reading spec
  0001's `documents.jsonl` from its own versioned directory. `.gitignore`'s negation for
  `wikidata_cache.jsonl.gz` updated to match any version. See `data_versions.json` and
  AGENTS.md §4.
