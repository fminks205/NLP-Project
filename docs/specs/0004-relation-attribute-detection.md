# 0004 — Relation attribute span detection

| | |
|---|---|
| **Status** | Accepted |
| **Depends on** | 0002 |
| **Superseded by** | — |
| **Owner** | Falk Minks |
| **Last updated** | 2026-09-28 |

## Context

Spec 0003 tried to recover relation type from one whole-sentence embedding per candidate
sentence. Findings
[0001](../findings/0001-relation-clustering-entity-and-template-bias.md) and
[0002](../findings/0002-entity-masking-effect-on-relation-clustering.md) show the same
failure mode repeating at every scale it was patched: whatever a sentence shares with
others dominates the embedding, whether or not it's the actual relation — first shared
person names (fixed by spec 0003 §2a's masking), then shared place/institution/topic once
person-masking was in place (the Harwell / crystal-growth clusters that prompted this
spec: multiple real sentences that cohere on a shared place or topic, not a shared kind of
social relation).

Rather than mask more entity types and keep clustering whole sentences, spec 0003 is
superseded by this spec and spec 0005: pull a sentence's individual factors — when, where,
at what institution, doing what — out as separately detected, typed spans, and only type
the piece that's actually about the relation (spec 0005's job, on the `action` span this
spec produces). This spec is the detection half.

Candidate sentence/pair finding is **not** revisited here — it stays exactly as spec 0002
built it (wikilink-derived). Real prose is unlikely to link enough of a relation's
surrounding factors in one sentence for links to serve as a general detector of them, so
links remain useful for finding candidate sentences but are not used to detect or validate
what's inside them — a deliberate narrowing, given the longer-term goal of running this
pipeline against corpora with no such links to lean on at all.

## Goal

Every unique candidate sentence (spec 0002's `candidates.jsonl`, deduplicated by
`(doc_id, section_idx, sent_idx)` — same key spec 0003 used) gets zero or more typed
attribute spans: `time`, `place`, `institution`, `action`. Deterministic and reproducible,
CPU-only.

## Non-goals

- **Detecting `topic`.** Raised during discussion (it's what "crystal growth" and similar
  clusters actually cohered on) and dropped from this first pass — no reliable text-only
  signal identified yet. Revisit once spec 0005's low-coverage output shows it's needed.
- **Re-detecting people.** Participants are already known from spec 0002's linked
  mentions; this spec detects what's *around* them, not the people themselves.
- **Validating detected spans against wikilinks.** An earlier draft of this spec proposed
  scoring NER precision/recall against link-derived types as a portability check. Dropped
  on review — out of scope for now, not attempted here.
- **Typing or clustering the `action` span's semantic content** (mentorship vs.
  collaboration vs. rivalry, etc.). This spec extracts the span; a follow-on spec types it
  — the same deferral spec 0003 originally made for the physics label set, now applied to
  a narrower `action` string instead of a whole-sentence embedding.

## Decision

### Detector: spaCy, same pinned model as spec 0002, more of its pipeline enabled

Spec 0002 runs `en_core_web_sm` with only `senter` enabled. This spec enables `ner` (for
`time` → `DATE`; `place` → `GPE`/`LOC`/`FAC`; `institution` → `ORG`) and `parser` (for
`action`, below) on the same pinned model — no new dependency, but real throughput cost on
43k+ sentences that spec 0002 never had to pay, unmeasured until run (acceptance
criteria).

### `action`: shortest dependency path between the two participant mentions

Participant spans are already known (spec 0002's `mentions.jsonl`), converted from
section-relative to sentence-local exactly as spec 0003 §2a's `build_sentence_starts` did
— that conversion logic is reused directly, not rebuilt. For each candidate pair, the
shortest dependency-parse path between the two participant tokens is found; the span it
covers (typically the governing verb phrase) is the `action` value. This is the piece most
likely to need iteration: expect failures on non-verbal or elliptical relations (e.g. "X,
Y's son, ...").

### `time` / `place` / `institution`: direct NER labels

`DATE` → `time`; `GPE`/`LOC`/`FAC` → `place`; `ORG` → `institution`. A sentence may
produce several spans of the same type; all are kept, not deduplicated to one per type.

### Pronoun-resolution fallback for the subject participant (added 2026-09-28)

A full-corpus run and diagnostic (see Changelog) found that `locate_participant_span`'s
name-match failing accounts for **95.6% of all `action`-detection misses — 42.4% of
every candidate pair in the corpus**, not a minor edge case. Every sampled failure was
the same pattern: a `subject_link` candidate whose subject is referred to only by
pronoun in that sentence ("He did doctoral research under...", never naming the subject
directly). This is spec 0002's already-documented coreference gap (§Non-goals),
previously unquantified.

**Accepted-for-now fix:** when name-matching fails for the subject, assume the earliest
third-person pronoun in the sentence refers to the subject (gender-appropriate where
`entities.jsonl`'s `gender` field is known; otherwise the earliest match of either set).
`locate_participant_span` now returns `(start, end, method)`, `method` ∈
`{"span", "name", "pronoun"}`, so every caller can tell a heuristic resolution from a
verified one.

**This is explicitly flagged as a pitfall, not a validated fix** — a sentence can
pronoun-reference someone other than the subject (a mother, a spouse, a colleague
introduced earlier in the same sentence), and this heuristic has no way to tell.
Surfaced three ways so it can't be silently forgotten: (1) `attribute_spans.jsonl`'s
`action` rows carry `pronoun_resolved: bool`; (2) `detect-attributes`'s manifest reports
`n_action_pronoun_resolved`/`action_pronoun_resolved_fraction`; (3) spec 0005's assembled
relation records carry the same flag and it renders in the human-readable summary. A
proper fix (real pronoun/coreference resolution, or at least "prefer the most recently
mentioned matching-gender person over a blanket subject assumption") is future work, not
attempted here — see Open questions.

## Interface

| Stage | Reads | Writes |
|---|---|---|
| `detect-attributes` | `candidates.jsonl`, `entities.jsonl`, `sentences.jsonl` (spec 0002) | `attribute_spans.jsonl` |

`mentions.jsonl` turned out not to be needed: `candidates.jsonl` already carries each
participant's section-relative span directly (`head_span`/`tail_span`, spec 0002's
`_candidate()`), so there is nothing left for `mentions.jsonl` to add — corrected here
during implementation, see Changelog.

**`attribute_spans.jsonl`** — one row per detected span:

```jsonc
{
  "sentence_id": "string — '<doc_id>:<section_idx>:<sent_idx>'",
  "candidate_id": "string|null — set for action (pair-scoped: depends on which two "
                  "participants), null for time/place/institution (sentence-scoped: "
                  "true regardless of which candidate pair asks)",
  "attr_type": "string — time | place | institution | action (open set — see spec 0005)",
  "value": "string — surface text of the span",
  "span": "[int, int] — sentence-local character offsets",
  "detector": "string — spacy_ner | dep_parse",
  "pronoun_resolved": "bool — action rows only; true if either participant was located "
                       "via the pronoun-fallback heuristic, not a verified name/span match"
}
```

**CLI:** `inpnet detect-attributes --in candidates.jsonl --mentions mentions.jsonl --entities entities.jsonl --sentences sentences.jsonl --out data/interim/attribute_spans/{version}/ --spacy-model en_core_web_sm`

## Alternatives considered

| Option | Why not (for now) |
|--------|-------------------|
| Validate spans against wikilink-derived types | Dropped on review — links are being deliberately narrowed to candidate-finding only; this would reintroduce them as a typing source. |
| LLM-based single-pass span extraction | More naturally open-ended/schemaless, but a new dependency class (API or local model) and a reproducibility cost (pinned version, temperature 0, cached raw output) not yet justified for a first pass. Revisit if spaCy NER/parse proves too weak. |
| Rule-based `action` extraction (verb-adjacent regex) | Cheaper than dependency parsing but doesn't generalize past a handful of sentence templates — spec 0003's own risk with fixed clusters. The parser is available in the same pinned model at one pipeline-config change. |

## Risks

- **`action` via shortest dependency path is unvalidated** — the biggest open risk here.
  Spec 0005's low-coverage output is the intended early-warning signal if this fails
  broadly.
- **NER precision on domain terms.** `en_core_web_sm` isn't tuned for scientific
  institution/place names; misses and mistyped spans are expected — not systematically
  measured (would need a hand-labelled sample), though spot-checks of the full run's
  output read as plausible.
- ~~Throughput~~ — resolved: full corpus in a single `nlp.pipe(batch_size=200)` pass, no
  additional batching needed.
- **The pronoun-resolution fallback (added 2026-09-28) is an unvalidated heuristic**,
  not a fix — "every pronoun refers to the subject" is wrong whenever a sentence
  pronoun-references someone else. Flagged in three places (row field, manifest counts,
  relation-record flag — see the Decision section above) specifically so this doesn't
  get mistaken for solved. No measurement yet of how often it's actually wrong (would
  need a hand-labelled sample); only how often it's *used* is currently tracked.

## Acceptance criteria

- [x] Runs on the full candidate-sentence set; produces `attribute_spans.jsonl` plus a
      manifest (model version, pipeline components enabled, per-`attr_type` counts).
      43,664 unique sentences, 76,205 candidate pairs → 131,419 spans.
- [x] Determinism: two runs on unchanged input produce byte-identical output
      (AGENTS.md §5). Verified on the full corpus — `attribute_spans.jsonl` SHA-256
      matched across two full runs.
- [x] Per-`attr_type` span count and a spot-check read of a sample (same style as spec
      0003's Eyring check) reported before spec 0005 consumes this output. Counts:
      institution 41,541, time 28,702, place 18,743, action 42,433 (found for 55.7% of
      candidate pairs). See `docs/methodology.md` §5.4 and `README.md` §8.
- [x] Unit tests: sentence-local span conversion (reusing spec 0003 §2a's tested logic),
      one fixture per `attr_type`, dependency-path extraction on a small hand-picked set
      of sentences. `tests/test_attributes.py`, 12 tests.

## Open questions

- Exact dependency-path-to-span rule (path tokens only vs. full governing subtree) —
  needs real examples to settle, not decided here. Partial evidence from the full run:
  the enclosing-span choice (rather than path-tokens-only) produces `action` spans
  averaging 91 characters against a ~187-character average sentence — often close to
  half the sentence, broader than intended, and can overlap already-typed
  `time`/`place`/`institution` spans. Worth revisiting, not yet decided.
- ~~Whether `parser` throughput forces batching changes not needed by spec 0002~~ —
  resolved: the full 43,664-sentence run completed without needing anything beyond the
  `batch_size=200` already used, well within the same order of magnitude as spec 0003's
  embedding run.
- **Correction to the line above** (originally written from two hand-picked examples,
  before running a real diagnostic — leaving the wrong guess visible rather than
  quietly editing it away): `action` recall being 55.7% is *not* mostly non-verbal
  relations. A full diagnostic over all 33,758 action-missing candidates
  (`n_candidates=76,205`) found: **95.6% (32,275, 42.4% of every candidate pair)**
  traced to the subject being pronoun-only in-sentence — addressed by the
  pronoun-resolution fallback above, itself an unvalidated heuristic, not a real fix.
  Only **4.3% (1,452, 1.9% of all candidates)** had both participants resolved but no
  dependency path — *that* bucket is the genuinely non-verbal/elliptical case this line
  originally described, and reading a sample confirms most of it really is appositive
  family-relation constructions ("his son, the diplomat...", "nephew of...") as
  expected, plus at least one real bug: `"...to Charles I. He was also..."` is two real
  sentences merged into one segment because spec 0002's senter doesn't treat `"Charles
  I."` (capital + roman numeral + period) as a sentence boundary — the same
  abbreviation-ambiguity class as `"Dr."`/`"Ph.D."` it already handles, not yet fixed.
  The remaining ~0.1% (31 candidates) trace to sentences truncated mid-wikimarkup at a
  stray `"["` — a spec 0001/0002 cleaning edge case, tiny in volume.
- **Real pronoun/coreference resolution** (beyond the blanket-subject heuristic above)
  is the highest-leverage remaining improvement, given it gates 42.4% of all candidates
  — not scoped here; a candidate for its own follow-on spec.
- **Two small, not-yet-fixed spec 0001/0002 bugs found via this diagnostic**, noted here
  since they surfaced through spec 0004's work, not spec 0001/0002's own: (1) sentences
  occasionally truncated mid-wikimarkup at a stray `"["` (~31 candidates observed); (2)
  the senter doesn't split on `"<CapitalLetter> <RomanNumeral>."` (e.g. `"Charles I."`),
  merging two real sentences into one segment. Both low-volume; low priority relative to
  the pronoun-resolution gap, but real.

## References

- spaCy dependency parsing and NER documentation.
- Reuses sentence-local span-conversion logic from spec 0003 §Decision 2a (spec
  superseded, code retained).

## Changelog

- 2026-09-28 — created, superseding spec 0003's clustering approach for relation
  structure. Scope narrowed from an earlier draft during review: dropped NER-vs-wikilink
  validation, dropped changes to `mentions.jsonl`'s schema, dropped `topic` detection.
- 2026-09-28 — accepted; implementation started
  (`src/inpnet/relations/attributes.py`, `inpnet detect-attributes`). Two corrections
  found while implementing: `mentions.jsonl` dropped as an input (redundant with
  `candidates.jsonl`'s own `head_span`/`tail_span`); `attribute_spans.jsonl` gained a
  `candidate_id` field, since `action` is pair-scoped (which two participants) while
  `time`/`place`/`institution` are sentence-scoped — the original schema only had
  `sentence_id`, which can't represent that distinction when one sentence produces
  several candidate pairs.
- 2026-09-28 — full-corpus run: 43,664 unique sentences, 76,205 candidate pairs →
  131,419 attribute spans (institution 41,541, time 28,702, place 18,743, action
  42,433); `action` found for 55.7% of candidate pairs. Determinism verified: rerun
  produced a byte-identical `attribute_spans.jsonl` (SHA-256 match). See
  `docs/methodology.md` §5.4.
- 2026-09-28 — diagnostic run over all 33,758 action-missing candidates (dev script,
  not a pipeline stage) traced 95.6% of misses to one cause: pronoun-only subject
  reference. Added the pronoun-resolution fallback (see Decision section above) and
  `pronoun_resolved` tracking on `action` rows, manifest counts, and (via spec 0005)
  the assembled relation record. Layer version bumped `0.0.1` → `0.0.2` in
  `data_versions.json` — a real methodological decision, not a bug fix; `0.0.1`'s
  output stays in place for comparison. Full-corpus rerun pending.
