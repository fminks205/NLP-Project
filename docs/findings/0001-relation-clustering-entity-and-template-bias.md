# 0001 — Entity and template bias in the first relation-clustering pass

| | |
|---|---|
| **From** | spec 0003, `cluster` stage |
| **Run** | `clustering_run_id 57eea9349750` — cosine metric, `cluster_selection_method=leaf`, `n_neighbors=15`, `n_components=15`, `min_cluster_size=5` |
| **Basis** | Informal single-reviewer read of a sample of `data/interim/cluster_summary.txt` (rendered via `inpnet cluster-summary`), not the two-annotator coherence review spec 0003 §Decision 5 still requires |
| **Owner** | Falk Minks |
| **Date** | 2026-08-26 |

## What this is

This full-corpus run produced 1,286 clusters over 43,664 sentences (27,881 noise, 63.8%;
see spec 0003's Changelog for how the run settings were chosen). A first manual read of
the output surfaced two related but distinct duplication patterns, plus a broader
observation about what the clustering is actually keying on. Recorded here, ahead of the
formal review, because they bear on how much weight the paper can put on unsupervised
clustering as a stand-in for relation type.

## Observation 1: verbatim boilerplate about a third party

**Cluster 4** (size 9) is near-identical text repeated across the corpus:

> "Paul Harteck was director of the physical chemistry department at the University of
> Hamburg and an advisor to the Heereswaffenamt (HWA, Army Ordnance Office)."

Checked against `relation_clusters.jsonl`: the cluster's 9 sentences come from **9
distinct documents** (Q104985, Q11210655, Q113727, Q5550685, Q7310378, Q76797,
Q8002172, Q94038, Q99622) — this is not a segmentation bug repeating one document's
sentence, it is the same descriptive sentence about Harteck copied across the Wikipedia
articles of (presumably) his students and colleagues, to give readers context about who
he was. The actual relation between Harteck and each article's own subject — that this
person worked under him, or was a member of his group — is a small fragment buried in
otherwise identical context, and appears in only one of the four sampled exemplars
("he was a member of Paul Harteck's group at the University of Hamburg").

## Observation 2: templated near-duplicates that each assert a real, distinct fact

**Cluster 36** (size 26) looks superficially similar — a repeated template — but reads
differently once you look at what varies:

> "Smoot was one of the 20 American recipients of the Nobel Prize in Physics to sign a
> letter …" / "Glashow is one of the 20 American recipients …" / "Politzer is one of the
> 20 American recipients …"

Checked the same way: 26 distinct documents. But here the template's slot is filled with
a **different person each time**, and each sentence is that article's own subject — so
each instance asserts a genuine, distinct fact (this specific person co-signed this
letter), not a copy-pasted fact about someone else. The surface repetition comes from
Wikipedia editors reusing the same sentence template for a real, structurally repeated
situation (many people did the same describable thing), not from irrelevant boilerplate.

**The distinction:** cluster 4 is boilerplate *about a third party*, where the relation
to the current article's subject is incidental to most of the repeated text. Cluster 36
is a boilerplate *template*, where the relation is the whole point of every instance and
only the named participant changes. Both produce near-identical sentences and both
therefore cluster tightly — but only one of them is duplication in the sense of "the
same fact asserted redundantly."

## Observation 3: clustering often tracks named entities, not relation structure

Broader pattern from the same read: clusters frequently group around a **shared named
entity** (same person mentioned in every exemplar) rather than a shared **relation type**
regardless of who's involved. Cluster 4 above is one instance — every exemplar is about
Harteck specifically. Cluster 894 from the same run is another: every exemplar names
Ernest Rutherford ("studied under Sir Ernest Rutherford," "worked with Ernest Rutherford
at the Cavendish Laboratory," "visited by Ernest Rutherford"), mixing what would likely
be different relation types (mentee, collaborator, visitor) under one cluster because
they share a participant, not a relation.

This isn't universal — cluster 806 from the same run ("his doctoral students include …")
is the counter-example: every exemplar names a *different*, unnamed advisor and
different students, and the cluster holds together on sentence structure/relation type
alone. So the run produces both kinds of cluster, and telling which is which — a genuine
relation-type cluster versus an entity-topic cluster that happens to cluster tightly for
an unrelated reason — is exactly what the (still-pending) formal coherence review needs
to sort out. This finding is that the distinction exists and showed up early, not a
measurement of how common either kind is; no count of entity-driven versus
relation-type-driven clusters has been done.

## Why this matters for the paper

It complicates treating unsupervised sentence clustering as a clean proxy for relation
type. Sentence-embedding similarity — as produced by the model and settings spec 0003
chose — appears to respond to shared participants and shared surface template at least
as readily as to shared relation semantics. Reporting cluster counts or coherence rates
without naming this risks overstating what the clustering step demonstrates.

## Open questions

- No systematic count yet of how many clusters are entity/topic-driven versus
  relation-type-driven. Would need to go through (a sample of) the 1,286 clusters and
  tag each.
- Whether verbatim near-duplicate clusters like #4 should be flagged/collapsed before
  the formal review, so reviewers aren't spending review effort on boilerplate rather
  than genuine relation candidates — undecided, not attempted here.
- No proposed methodological fix is recorded here — this document is observational, not
  a decision. If a fix is worth trying (e.g. something to disentangle entity identity
  from relation content before embedding), that belongs in its own spec, weighed against
  the alternatives, not decided from two examples.
