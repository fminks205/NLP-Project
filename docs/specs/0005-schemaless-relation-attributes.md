# 0005 — Schemaless relation attributes

| | |
|---|---|
| **Status** | Accepted |
| **Depends on** | 0002, 0004 |
| **Superseded by** | — |
| **Owner** | Falk Minks |
| **Last updated** | 2026-09-28 |

## Context

Spec 0003 modeled a relation as one `(person1, person2, cluster_id)` triple per sentence —
a single label standing in for everything the sentence asserts. Collapsing distinct kinds
of information (when, where, at what institution, doing what) into one embedding is what
made spec 0003's clusters key on incidental shared context instead of relation type
(findings 0001, 0002; see spec 0003's Changelog for the supersession). It also doesn't
generalize past physics/Wikipedia's own conventions, given the project's longer-term goal
of running against corpora that have no wikilink crutch to fall back on at all.

This spec replaces the triple with a relation record that's a bag of independently typed
attributes, assembled from spec 0002's participants and spec 0004's detected spans.
"Schemaless" here means the *set of attribute types* is open and can vary by domain — the
record container itself is fixed (this spec); what fills it isn't.

## Goal

Every candidate pair gets one relation record: its two participants, plus whatever
`time`/`place`/`institution`/`action` attributes spec 0004 found in its sentence.
Sentences where detection covers too little of the sentence to be a trustworthy record are
separated out for review, not silently kept or silently dropped.

## Non-goals

- **Typing the `action` attribute's semantic content** (mentorship vs. collaboration vs.
  rivalry, etc.). This spec assembles the record; typing it is a follow-on spec — the same
  deferral spec 0003 made for the physics label set, now applied to a narrower `action`
  string instead of a whole-sentence embedding.
- **Adding `topic` as an attribute type.** Spec 0004 doesn't detect it yet; out of scope
  until it does.
- **Multi-relation-per-sentence disambiguation.** Spec 0003's known limitation (one
  sentence naming 3+ people collapses to a single cluster/record) is inherited unchanged
  — still not solved here.
- **Deleting spec 0003's clustering code.** Superseded per `docs/specs/README.md`, not
  removed — whether `src/inpnet/relations/cluster.py` stays as an ablation baseline or is
  later removed is a separate call, not made here.

## Decision

### Relation record: participants + open attribute list

One record per candidate pair (spec 0002's `candidates.jsonl` rows — not deduplicated to
unique sentences the way spec 0003/0004 are, since the two participants are part of the
record's identity even when the sentence is shared by several pairs).

### Coverage metric and the low-coverage split

`coverage` = fraction of the sentence's content tokens (non-stopword, non-punctuation,
read off spaCy's own token attributes already available from spec 0004's parse) that fall
inside a participant span or any detected attribute span. Records below `--min-coverage`
(default `0.5` — a proposal to tune against the first run's actual distribution, not a
measured result) go to `low_coverage_relations.jsonl` instead of `relations.jsonl`. This is
exactly the use spec 0004 flagged as its biggest risk for: reviewing this file shows
whether `action` extraction is failing on sentences it should have caught, or whether a
genuinely new `attr_type` is missing from spec 0004's set entirely.

## Interface

| Stage | Reads | Writes |
|---|---|---|
| `assemble-relations` | `candidates.jsonl`, `entities.jsonl`, `sentences.jsonl` (spec 0002), `attribute_spans.jsonl` (spec 0004) | `relations.jsonl`, `low_coverage_relations.jsonl`, plus a human-readable rendering of each (spec 0003's `cluster-summary` precedent) |

`mentions.jsonl` dropped as an input, same correction spec 0004 made — `candidates.jsonl`
already carries what's needed. `sentences.jsonl` added, for the same section-relative →
sentence-local span conversion spec 0004 needs (`build_sentence_starts`).

**`relations.jsonl`** / **`low_coverage_relations.jsonl`** — same schema, split by
`coverage`:

```jsonc
{
  "relation_id": "string — spec 0002's own candidate_id, already deterministic per pair; no new hash minted",
  "sentence_id": "string",
  "sentence": "string — the original sentence text. Not previously in this schema; added during implementation because a human-readable rendering (this spec's Goal) needs the source text, and AGENTS.md §5 requires provenance back to source",
  "participants": [{"mention_id": "string", "qid": "string|null"}],
  "attributes": [
    {
      "attr_type": "string — open set, e.g. time | place | institution | action",
      "value": "string",
      "span": "[int, int]",
      "source": "string — spacy_ner | dep_parse"
    }
  ],
  "coverage": "float",
  "pronoun_resolved": "bool — true if either participant's position in the sentence was "
                       "located via spec 0004's pronoun-fallback heuristic rather than a "
                       "verified name/span match (added 2026-09-28 — see spec 0004's "
                       "Decision section on why); lets a reviewer filter to exactly the "
                       "records resting on that unvalidated assumption"
}
```

**Human-readable rendering** (`inpnet relations-summary`, mirrors spec 0003's
`cluster-summary`): for each record, the sentence with every detected attribute shown
inline, and a merged, readable rendering of the token runs `coverage` did *not* count as
covered. The same renderer serves both `relations.jsonl` ("what did we type") and
`low_coverage_relations.jsonl` ("what's left over — review whether spec 0004's detectors
or its `attr_type` set need work"), since both share this schema. `assemble-relations`
writes this rendering automatically alongside the JSONL, not only on separate request —
the Goal below is explicit that unclassified content must be visible, not just present in
a field nobody reads by default.

**CLI:** `inpnet assemble-relations --in candidates.jsonl --entities entities.jsonl --sentences sentences.jsonl --attributes attribute_spans.jsonl --out data/interim/relations/{version}/ --min-coverage 0.5`

## Alternatives considered

| Option | Why not (for now) |
|--------|-------------------|
| Fixed columns (`time`, `place`, `institution`, `action` as named top-level fields) | Simpler to query but not extensible — a new domain's attribute type would need a schema migration, exactly what "transferable to other domains" was meant to avoid. |
| Drop low-coverage sentences instead of writing them out | Loses exactly the signal needed to iterate on spec 0004's detectors; explicitly what was asked for during design. |
| Coverage measured in characters, not content tokens | Skews toward long spans regardless of relevance (e.g. a 40-character institution name would dominate a 60-character sentence); token-based tracks "how much of what's asserted was actually captured" more directly. |

## Risks

- **`--min-coverage` is arbitrary** until the first run's distribution is seen — same
  posture spec 0003 took with its own untuned hyperparameters. Measured on the full
  corpus: mean coverage 0.510 with the default `0.5` threshold, so roughly half of
  candidate pairs sit right around the cutoff — the threshold has real leverage over the
  split and is worth deliberately tuning next, not left at its default.
- **Low-coverage volume is large** (measured: 50.4%, 38,376 of 76,205), echoing spec
  0003's 66% noise bucket — reported as its own number per this Risk, not treated as a
  failure by itself. A sample read confirms it's doing its job (see Acceptance
  criteria), not hiding a broken detector as noise.
- ~~`relation_id` stability across reruns~~ — resolved during implementation by reusing
  spec 0002's own `candidate_id` directly (see Changelog) instead of minting a new id.

## Acceptance criteria

- [x] Runs on the full candidate set; produces both output files plus a manifest
      (`--min-coverage` used, record counts, coverage distribution). 76,205 candidate
      pairs → 37,829 relations, 38,376 low-coverage (50.4%), mean coverage 0.510.
- [x] Determinism: byte-identical rerun on unchanged input (AGENTS.md §5). Verified on
      the full corpus — `relations.jsonl` and `low_coverage_relations.jsonl` both
      SHA-256-matched across two full runs (through a rerun of `detect-attributes` too,
      so the whole 0004→0005 chain is confirmed deterministic end to end).
- [x] Manual read of a sample of `low_coverage_relations.jsonl` (spot-check style, as
      spec 0003 did for clusters) confirming it actually surfaces detector gaps rather
      than just short sentences. Confirmed: sampled records mostly show a correctly
      detected `institution`/`time` alongside a missing `action` (e.g. "supervised",
      "co-authored" not found) — real detector gaps, not short/garbled sentences. See
      `docs/methodology.md` §5.4.
- [x] Unit tests: coverage computation on hand-built fixtures, the low-coverage split
      boundary, `relation_id` determinism across two runs. `tests/test_assemble.py`, 7
      tests.

## Open questions

- What `--min-coverage` value the first real run's distribution actually supports — open
  until measured.
- Whether `attributes` should allow multiple values of the same `attr_type` per record
  (e.g. two `place` spans) — the list-shaped schema allows it implicitly; not yet tested
  against a real multi-place sentence.

## References

- Supersedes spec 0003's relation representation; reuses spec 0002's participant/mention
  data unchanged; consumes spec 0004's attribute spans directly.

## Changelog

- 2026-09-28 — created, superseding spec 0003.
- 2026-09-28 — accepted; implementation started (`src/inpnet/relations/assemble.py`,
  `inpnet assemble-relations`, `inpnet relations-summary`). Corrections found while
  implementing: `mentions.jsonl` dropped as an input (see spec 0004's identical
  correction), `sentences.jsonl` added; `relation_id` set to spec 0002's own
  `candidate_id` rather than minting a new hash (already deterministic, one fewer thing
  to get wrong); `sentence` added to the record schema, required for the human-readable
  rendering this spec's Goal calls for; `relation_id` risk item resolved by that same
  choice.
- 2026-09-28 — full-corpus run: 76,205 candidate pairs → 37,829 relations, 38,376
  low-coverage (50.4%) at the default `--min-coverage 0.5`, mean coverage 0.510.
  Determinism verified end to end (0004 → 0005 rerun, byte-identical). Low-coverage
  sample read: mostly missing `action` alongside correctly-typed `time`/`institution`,
  confirming the split surfaces spec 0004's `action`-recall gap rather than noise. See
  `docs/methodology.md` §5.4.
- 2026-09-28 — `pronoun_resolved` added to the record schema, following spec 0004's
  pronoun-fallback heuristic (a diagnostic found 42.4% of all candidate pairs have a
  pronoun-only subject in-sentence — see spec 0004's Decision section). Layer version
  bumped `0.0.1` → `0.0.2` in `data_versions.json`: this is a real methodological
  decision (what counts as a located participant changed), not a bug fix to what
  `0.0.1` was already supposed to mean — same posture spec 0003 took bumping to its own
  `0.0.2` for entity masking. `0.0.1`'s output is left in place for comparison.
