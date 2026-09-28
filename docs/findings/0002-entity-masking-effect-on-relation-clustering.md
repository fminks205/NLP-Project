# 0002 — Measured effect of entity masking on the full-corpus clustering run

| | |
|---|---|
| **From** | spec 0003 §Decision 2a, `cluster` stage |
| **Runs compared** | `0.0.1` unmasked (`clustering_run_id 57eea9349750`) vs `0.0.2` masked (`clustering_run_id 28755104d83f`) — identical hyperparameters (`cosine` metric, `cluster_selection_method=leaf`, `n_neighbors=15`, `n_components=15`, `min_cluster_size=5`); masking is the only variable that changed |
| **Basis** | Corpus-level counts from both manifests, plus a targeted check against the Henry B. Eyring example that motivated §Decision 2a |
| **Owner** | Falk Minks |
| **Date** | 2026-09-28 |

> **Superseded by the addendum below (same date).** The `0.0.2` run this note originally
> analyzed had a masking bug (link-mention spans sliced with the wrong offset base —
> see spec 0003 Changelog) that left most linked-person names unmasked. Everything below
> this notice describes that buggy run; it's kept as the historical record of how the bug
> was found (a real, reproducible observation at the time), not as the current state of
> `0.0.2`. See **Addendum: corrected run** at the bottom for what actually holds once the
> bug was fixed.

## What this is

Spec 0003 §Decision 2a added entity masking (every detected person mention replaced with
a single `[PERSON]` placeholder before embedding) to address a bias
[finding 0001](0001-relation-clustering-entity-and-template-bias.md) Observation 3 had
already flagged: clusters frequently track a shared named entity rather than a shared
relation type. This note records what actually changed once the full corpus was
re-clustered with masking on, checked directly against the concrete example that framed
the decision, rather than assuming the fix worked.

## Corpus-level effect

| | `0.0.1` (unmasked) | `0.0.2` (masked) |
|---|---|---|
| Clusters | 1,286 | 1,224 |
| Noise | 27,881 (63.8%) | 28,862 (66.1%) |
| Sentences with a mention masked | — | 43,664 / 43,664 (100%) |

Every sentence had at least one detected person mention (expected: spec 0002's candidate
pairs are sentences that already contain a person mention by construction). Masking
measurably shifted cluster structure — fewer clusters, more noise — without a dramatic
reshuffle. Slightly *more* noise is a plausible cost of removing a signal (surface
identity) some clusters had genuinely been keying on: a sentence that only cohered with
others because it named the same person may no longer cohere with anything once that
name is gone.

## The Eyring example, checked directly

The three sentences quoted in spec 0003 §Decision 2a:

> "Following the death of church president Howard W. Hunter, Eyring was sustained as a
> member of the church's Quorum of the Twelve Apostles..."
> "Eyring served as president of Ricks College from 1971 to 1977, as a counselor to
> Presiding Bishop Robert D. Hales..."
> "Eyring has served twice as commissioner of church education..."

**Landed in the same cluster in both runs** (cluster 10 unmasked, cluster 57 masked).
Masking did not split them apart. But the cluster's *composition* did change:

- **Unmasked, cluster 10 is exactly 7 sentences, all from Henry B. Eyring's own article
  (Q949425) and no other document.** A textbook entity-driven cluster.
- **Masked, cluster 57 is the same 7 Eyring sentences plus one sentence from a different
  article** (Q61951207): "After graduating from law school, Meserve served as a law
  clerk to Justice Benjamin Kaplan of the Massachusetts Supreme Judicial Court during
  1975-76 and Justice Harry Blackmun of the Supreme Court of the United States during
  its 1976-77 term." Masked, this becomes the same "served as [role] to/of
  [institution]... during [dates]" template as the Eyring sentences — a different
  person, same career-summary phrasing.

## Reading this honestly

This is evidence masking is doing *something* real — an unrelated document's sentence
now qualifies for a cluster it previously couldn't reach, because the thing distinguishing
it from Eyring's sentences (the name) is gone and what's left is genuine template
similarity. But it is not evidence the bias is fixed. Two explanations for why the
original three still cluster together are both consistent with what was measured here,
and this run cannot distinguish them:

1. They still share "Eyring" indirectly — via document-level artifacts masking doesn't
   remove (the same author's prose style, the same article's terminology, proper nouns
   *near* the masked name that survive, like "church," "commissioner," "Quorum").
2. They genuinely share a relation-type template — Wikipedia's convention for
   summarizing someone's institutional career ("served as X ... from Y to Z") — and a
   coherent cluster of that template legitimately includes multiple sentences from the
   same person's own career alongside sentences from others', because career-summary
   sentences from the same article naturally reuse the same construction.

Distinguishing these needs more than one example — either the pending two-annotator
coherence review (spec 0003 §Decision 5) reading a sample of `0.0.2`'s clusters with an
eye specifically for single-document dominance, or a systematic per-cluster count of
distinct source documents (not done here or in
[finding 0001](0001-relation-clustering-entity-and-template-bias.md)).

## Open questions

- **Per-cluster document diversity, systematically measured.** This note checked one
  cluster by hand. A corpus-wide "fraction of each cluster's sentences drawn from a
  single document" statistic, compared `0.0.1` vs `0.0.2`, would say whether cluster 57
  is typical of the masked run's improvement or an outlier.
- **Determinism (byte-identical rerun) for `0.0.2`** — not checked here, same open
  acceptance criterion carried over from `0.0.1`.
- **The two-annotator coherence review** (spec 0003 §Decision 5) has not run against
  either version yet; cluster counts from both remain provisional.
- No proposed follow-up fix is recorded here — if the residual entity-adjacency
  explanation (item 1 above) turns out to dominate, a next step might mask other
  proper nouns (institutions, organizations) too, not just people — but that is a new
  decision to weigh against alternatives, not assumed here.

## Addendum: corrected run (2026-09-28, same day)

A user check against a *different* concrete example — "Sir Humphry Davy," mentioned as
the linked (non-subject) party in three unrelated articles, still clustering together
after masking — found that link-mention masking wasn't actually working. Root cause:
`build_entity_index` sliced a mention's `start`/`end` directly against sentence-local
text, but those offsets are section-relative (spec 0002 §Decision 4). For a real mention
span like `[358, 374]` against a 124-character sentence, Python's slicing silently
clipped instead of erroring — the placeholder was appended at the very end, and the name
was never actually removed. This affected every linked-person mask; only the article
subject (matched by name, not offset) was masked correctly, which is exactly why the
Eyring case above (subject-only, no other linked person in those sentences) looked like
a "partial" result rather than a total failure — the one thing that *did* work in the
buggy run happened to be the only mechanism the Eyring example exercised.

Fixed (see spec 0003 Changelog) by converting each link span to sentence-local using
that sentence's own section-relative start from `sentences.jsonl`, plus a bounds check
that drops any still-invalid span rather than silently corrupting text. Full corpus
re-clustered, same hyperparameters, same `clustering_run_id` (config-derived, so an
unrelated code fix doesn't change it — a real limitation of that scheme, not something
fixed here):

| | `0.0.1` unmasked | `0.0.2` buggy masking | `0.0.2` corrected masking |
|---|---|---|---|
| Clusters | 1,286 | 1,224 | 1,232 |
| Noise | 27,881 (63.8%) | 28,862 (66.1%) | 28,542 (65.4%) |
| Sentences actually changed by masking | — | 43,664 / 43,664 (100%, vacuously — every sentence counted regardless of whether anything was removed) | 43,648 / 43,664 (99.96%) |

**Re-checked both motivating examples directly.** All three Eyring sentences (cluster
57 in the buggy run) and all three "Sir Humphry Davy" sentences from the bug report are
now `cluster_id -1` (noise) in the corrected run — **not clustered with each other, or
with anything else.** This is a materially different result from the buggy run's
"masking nudged the cluster wider but didn't split it" reading above: with linked
mentions actually masked, these particular sentences no longer share enough content to
cluster with anything at all, which is a much more direct confirmation that removing the
name was what was previously holding them together.

This doesn't retroactively answer the open question in the original note (whether
`0.0.2` clusters in general lean template-driven vs. still entity-adjacent) — that still
needs the systematic per-cluster document-diversity check, now against the corrected
run instead of the buggy one. It does resolve the concrete case this note was built
around.
