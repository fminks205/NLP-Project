# 0003 — Relation typology

| | |
|---|---|
| **Status** | Superseded |
| **Depends on** | 0002 |
| **Superseded by** | 0004, 0005 |
| **Owner** | Falk Minks |
| **Last updated** | 2026-08-26 |

## Context

Spec 0002 produced 76,205 candidate person-pairs, each carrying the sentence that put the
two people together. Nothing about *what kind of relation* the sentence asserts exists
yet.

The pipeline built through specs 0001–0002 is not physics-specific anywhere — seed
selection and mention resolution both operate on "a Wikipedia biography," not "a
physicist's biography." The **relation typology is the one place domain assumptions get
baked in**, and physics is a poor template for that: physicist biographies are
comparatively fact-based, so their interpersonal prose leans collaboration/mentorship and
rarely encodes real antagonism, while a typology built to fit that skew would misfit
fields where dispute and disagreement are central (politics, literature, criticism). A
keyword pass over `candidates.jsonl` bears this out — antagonism cues here are both rare
(order of a thousand) and noisy: "opposed Donald Trump in the 2016 … election" matches an
antagonism keyword list but asserts no interpersonal relation at all.

This spec therefore defines a **field-agnostic intermediate layer** — candidate sentences
grouped by an unsupervised clustering step, with no relation labels presupposed — and the
**contract** a domain-specific typology must satisfy to consume that layer. The concrete
label set for physics is deferred to a follow-on spec, so it can be revised without
touching the clustering stage, and so a different research purpose can plug a different
label set into the same clusters.

## Goal

Two things are true once this is done:

1. Every candidate pair's sentence is embedded and assigned to a cluster, deterministically
   and reproducibly, with cluster coherence checked by human review rather than assumed.
2. A documented, versioned contract exists for how a domain-specific relation typology plugs
   into these clusters, so the annotation guide, any model prompt, and the evaluation script
   all read the same definition and cannot drift apart.

## Non-goals

- **The physics-specific relation label set** (advisor, spouse, rival, co-author, …) — this
  is the biggest thing deferred. It becomes its own spec once this interface exists, and it
  is the project's first *plug-in* against this contract, not part of it.
- **Automatic cluster naming.** Clusters get a human-assigned interpretive label during
  review; nothing here infers what a cluster "means."
- **Full relation annotation** — building a gold-labelled dataset is the follow-on spec's
  job, once there is a label set to annotate against.
- **Wikidata property mapping for specific relation types.** The contract *reserves a field*
  for it (so distant supervision and novelty measurement are possible later); which
  properties map to which physics-specific labels is decided in the follow-on spec.
- **Hedging and temporal scope** (a relation being asserted tentatively, or holding only for
  a period). These are annotation-guide concerns for whatever typology plugs in, not
  clustering-interface concerns — noted so they are not silently dropped, not resolved here.
- **GPU-dependent modeling.** Consistent with spec 0002 (no GPU today, one expected later),
  this spec picks a CPU-feasible embedding model, not the strongest available one.

## Decision

### 1. Unit of clustering: unique sentences, not pairs

76,205 candidate pairs resolve to **43,664 distinct sentences** (measured: grouping
`candidates.jsonl` by `(doc_id, section_idx, sent_idx)`) — some sentences produce several
pairs (three or more people in one sentence). Clustering runs once per unique sentence;
every candidate pair inherits its sentence's `cluster_id` by joining on that key. This means
the layer cannot yet distinguish two different relations asserted about two different pairs
in one sentence ("X married Y; both later collaborated with Z" would give X–Y and Y–Z and
X–Z the same cluster). That is the same limitation spec 0002 already flags for the subject
heuristic — recorded here as a risk, not solved.

### 2. Embedding: sentence-transformers, CPU-feasible model

Sentence embeddings are the natural representation for "group by what relation-like content
a sentence expresses" independent of surface wording ("his wife" and "married" should land
near each other; TF-IDF would keep them apart on lexical overlap alone). Proposed:
`sentence-transformers` with a small model (e.g. `all-MiniLM-L6-v2`, ~80 MB, CPU-only) —
this is a **proposal to validate, not a measured result**. `sentence-transformers` is not
currently a project dependency; this spec adds it (plus its CPU-only PyTorch backend), the
project's first ML dependency beyond spaCy's `senter`. Actual throughput on this machine and
cluster quality on this corpus are unmeasured — both are acceptance criteria below, not
assertions here.

### 2a. Entity masking before embedding

Unmasked, sentence embeddings key on **who** a sentence names as readily as on what
relation it asserts —
[`docs/findings/0001-relation-clustering-entity-and-template-bias.md`](../findings/0001-relation-clustering-entity-and-template-bias.md)
Observation 3 (the first full-corpus run's Harteck and Rutherford clusters group on a
shared participant, not a shared relation type) flagged exactly this and left "something
to disentangle entity identity from relation content before embedding" as an open
question. The article subject makes it worse: it is named in nearly every sentence of
its own article, so a document's own sentences pull toward each other on that shared name
alone. Concretely, three real sentences about Henry B. Eyring:

> "Following the death of church president Howard W. Hunter, Eyring was sustained as a
> member of the church's Quorum of the Twelve Apostles..."
> "Eyring served as president of Ricks College from 1971 to 1977, as a counselor to
> Presiding Bishop Robert D. Hales..."
> "Eyring has served twice as commissioner of church education..."

assert three different kinds of relation (a succession/ordination, an office held under a
superior, an appointment), but share little besides "Eyring" and generic biographical
phrasing — entity bias, not relation similarity.

Before embedding, every detected person mention in a sentence is replaced with a single
generic placeholder, `[PERSON]`:

- **Linked mentions** (`mentions.jsonl`, `mention_type=link`) carry exact character
  spans — **section-relative**, per spec 0002 §Decision 4, not sentence-relative, so
  masking converts each one to a sentence-local offset using that sentence's own start
  from `sentences.jsonl` before slicing (see Changelog: this conversion was missing in
  the first implementation, which silently masked nothing for most sentences) — and are
  masked directly, for *every* linked person in the sentence, not only the two in a
  given candidate pair, since a suppressed `link_link` enumeration (spec 0002 §Decision
  5) still leaves those names sitting in the sentence text.
- **The article subject** (`mention_type=subject`) has no span — Wikipedia never
  self-links — so it is matched by name instead: its full canonical name
  (`entities.jsonl`) and, since prose refers to it by surname after first mention, the
  surname alone, both as whole-word matches.

One shared placeholder, not a per-entity or per-role token, is a deliberate choice: the
goal is to remove identity, not to re-encode "how many distinct people" or "which one is
the subject" as a different structural signal for the clusterer to key on instead.

This makes `mentions.jsonl`, `entities.jsonl`, and `sentences.jsonl` (spec 0002 outputs)
new inputs to the `cluster` stage, alongside `candidates.jsonl` — see §Interface. Masking only affects what
`sentence_embeddings.npy` is built from; `relation_clusters.jsonl` and
`cluster_summary.jsonl` are unchanged in shape, and `cluster_summary.jsonl`'s
exemplar/sample sentences still show the **original**, unmasked text — masking is for
what the embedder sees, not for what a human reviewer reads during §Decision 5's
coherence check. The manifest records whether masking was applied
(`config.mask_entities`) and how many sentences it actually changed
(`counts.n_sentences_masked`); `--no-mask-entities` on `inpnet cluster` is kept as an
ablation switch, off by default.

**Known limitation:** subject-name matching is a whole-word string match on the
canonical name and surname, not coreference — it does not catch pronouns ("he," "she"),
which spec 0002 already scopes out as a coreference concern (§Non-goals). It can
under-mask a subject referred to by a nickname or spelling the Wikipedia title doesn't
carry, and could in principle over-mask if a surname coincided with an unrelated
capitalized word — not observed so far, but not proven absent either.

### 3. Clustering: density-based, not a fixed K

The "discover via clustering, no priors" direction rules out committing to a cluster count
upfront (k-means with a chosen K). Proposed: reduce dimensionality with UMAP, then cluster
with HDBSCAN (the embed → reduce → cluster pattern popularised by BERTopic). HDBSCAN's
practical advantage here is a built-in **noise label**: sentences that don't fit any dense
group are marked noise rather than forced into the nearest cluster. Given how much of
`candidates.jsonl` is weak co-occurrence rather than an asserted relation (spec 0002's own
risk section: the subject heuristic over-fires, and `people_in_sentence` enumerations were
already capped for the same reason), a noise bucket is expected to be large and is itself a
reportable number, not a failure of the method.

### 4. Determinism

Model version/revision pinned, sentence processing order fixed (sorted by sentence key),
fixed random seeds for UMAP and HDBSCAN, all recorded in the manifest. A rerun on unchanged
input must reproduce identical `cluster_id` assignments — same hard requirement as every
other stage (AGENTS.md §5).

**`cluster_id` is run-scoped, not a stable identifier.** HDBSCAN cluster numbering is an
implementation-detail ordering, not a persistent name. A typology plug-in's mapping from
clusters to labels is only valid against the specific clustering run it was built from; the
plug-in config records the `clustering_run_id` it targets (from that run's manifest).
Re-clustering after a code or data change requires re-deriving the mapping — see Risks.

### 5. Validation is a human read, not a metric alone

For each cluster: the exemplar sentences nearest its centroid, plus a random sample, go to
human review. Both available annotators (see below) independently read the same sample and
record a coherence verdict (does this cluster read as one relation-like theme, yes/no/mixed)
and, if yes, a free-text interpretive name. Raw agreement across the two annotators is
reported — not a formal kappa, since the review is a coherence judgment on a small sample of
clusters, not a per-item labeling task with a fixed label set. This follows the same
"manual read is the highest-value check" pattern spec 0002's acceptance criteria already
established.

Two annotators are available for this project, which also means the *next* spec — the
physics typology and its gold annotation — can measure real inter-annotator agreement
instead of relying on a documented guide alone. Annotation budget for that gold set is
still open; it depends on how many clusters emerge and is deferred to that spec.

### 6. The typology plug-in contract

A domain-specific typology is one config file (a `docs/specs/typologies/<name>.yaml`,
schema fixed by this spec) with one entry per relation type:

```yaml
type_id: string            # stable short id, e.g. "mentorship"
label: string               # human-readable name
definition: string          # what an annotator checks for
direction: directed | symmetric
cluster_ids: [int]           # which cluster_id(s) from a named clustering_run_id map here
clustering_run_id: string    # which run's clusters this mapping is valid against
wikidata_properties: [string]  # optional P-ids, for distant supervision / novelty measurement
examples: [string]           # sentences illustrating the type
```

This single file is meant to be the one thing the annotation guide, any model prompt, and
the evaluation script all read — generated from it, not hand-copied into three places. That
tooling (a guide renderer, a prompt builder, an eval-script loader) is follow-on work; this
spec fixes only the schema they will all consume, plus a validator: every `cluster_ids`
entry must exist in the named run's `cluster_summary.jsonl`, checked by a test using a toy
example (not the physics typology, which doesn't exist yet).

`cluster_ids` is a list, and one cluster may in principle be referenced by more than one
type's mapping in different plug-ins — clustering granularity and typology granularity are
not assumed to match. The `wikidata_properties` field is what makes distant supervision and
the novelty measurement (how much of the extracted graph has no Wikidata equivalent)
possible later: once a plug-in exists, an edge whose type maps to no property, or whose
property doesn't already hold for that pair in `wikidata_cache.jsonl`, counts as novel. That
computation is straightforward given the field; running it is the follow-on spec's job.

## Interface

One new stage.

| Stage | Reads | Writes |
|---|---|---|
| `cluster` | `candidates.jsonl`, `mentions.jsonl`, `entities.jsonl`, `sentences.jsonl` | `sentence_embeddings.npy`, `sentence_ids.jsonl`, `relation_clusters.jsonl`, `cluster_summary.jsonl` |

`mentions.jsonl`, `entities.jsonl`, and `sentences.jsonl` (spec 0002 outputs) drive
entity masking before embedding (§Decision 2a) — `sentences.jsonl` specifically for
converting a link mention's section-relative span to sentence-local; none of the three
are otherwise read for clustering.

**`sentence_ids.jsonl`** — row order matches `sentence_embeddings.npy`:

```jsonc
{ "sentence_id": "string — '<doc_id>:<section_idx>:<sent_idx>'" }
```

**`relation_clusters.jsonl`** — one row per unique sentence:

```jsonc
{
  "sentence_id": "string",
  "cluster_id": "int — -1 is HDBSCAN noise",
  "cluster_prob": "float — HDBSCAN membership probability"
}
```

**`cluster_summary.jsonl`** — one row per `cluster_id` (excluding noise):

```jsonc
{
  "cluster_id": "int",
  "clustering_run_id": "string",
  "size": "int",
  "exemplar_sentences": ["string — nearest to centroid"],
  "sample_sentences": ["string — random draw"],
  "coherence_reviewed_by": ["string — annotator ids"],
  "coherence_verdict": "enum|null — yes | no | mixed",
  "human_label": "string|null — free-text interpretation"
}
```

Candidate pairs join to a `cluster_id` via `(doc_id, section_idx, sent_idx)` → `sentence_id`
at read time; `candidates.jsonl` itself is not rewritten (spec 0002's stage boundary — no
stage overwrites another stage's output).

**CLI:** `inpnet cluster --in candidates.jsonl --mentions mentions.jsonl --entities entities.jsonl --sentences sentences.jsonl --out data/interim/relation_typology/{version}/ --model all-MiniLM-L6-v2 --min-cluster-size <n>`
(`--mentions`/`--entities`/`--sentences` default to the current entity_mention_layer
version's files; `--no-mask-entities` disables masking as an ablation switch.)

`{version}` is this layer's current version in [`data_versions.json`](../../data_versions.json)
at the repo root, bumped by hand; the CLI defaults already resolve it, so `--in`/`--out`
only need overriding to point at a non-current run. See AGENTS.md §4.

**Typology config schema:** `docs/specs/typologies/TEMPLATE.yaml`, per §Decision 6.

## Alternatives considered

| Option | Why not (for now) |
|--------|-------------------|
| Fixed physics-specific typology now | Doesn't generalize, and the methodology is explicitly meant to extend past physics. See Context. |
| Zero-shot relation extraction directly (e.g. GLiREL, referenced in spec 0002) | Needs a candidate label set as input — exactly the domain commitment being deferred. Revisit once a typology plug-in exists. |
| k-means with K chosen by elbow/silhouette | Requires committing to a cluster count upfront, which presupposes structure the "no priors" direction rules out. Kept as a fallback if HDBSCAN degenerates (e.g. one giant cluster plus all noise). |
| TF-IDF + cosine clustering, no new heavy dependency | Groups by lexical overlap, not relation semantics — "married" and "his wife" would separate despite asserting the same relation. Cheap enough to keep as a sanity-check baseline against the embedding clusters. |
| spaCy static vectors | `en_core_web_sm` (the pinned model) ships no word vectors — only `_md`/`_lg` do. Would need a model swap for no clear benefit over sentence-transformers. |

## Risks

- **Embedding model is unvalidated.** `all-MiniLM-L6-v2` is a proposal; if the human review
  finds clusters mostly incoherent, the fix is a different model or preprocessing, not a
  different label list — the acceptance criteria are written so that failure shows up here.
- **`min_cluster_size` and UMAP hyperparameters are arbitrary** until swept against the
  actual corpus. A single untuned setting could either merge distinct relation types into
  one cluster or shatter one relation type across many.
- **`cluster_id` instability across reruns.** Any typology plug-in mapping is tied to one
  `clustering_run_id`; re-clustering after a code/data/dependency change invalidates
  existing mappings until re-derived. This is a real maintenance cost the contract makes
  visible rather than hides.
- **Sentence-level granularity caps precision** at whatever spec 0002's candidate pairs
  already cap it at — this stage cannot fix multi-relation sentences, only cluster what it's
  given.
- **First ML dependency in the project.** `sentence-transformers` pulls in a CPU PyTorch
  build; install size/time should be checked against `uv sync` before this is treated as
  a small addition.

## Acceptance criteria

- [ ] `cluster` runs on the full corpus (43,664 unique sentences) and produces the four
      output files plus a manifest recording model version, seeds, and hyperparameters.
- [ ] Determinism: two consecutive runs on unchanged input produce byte-identical
      `relation_clusters.jsonl` (SHA-256 comparison), per AGENTS.md's determinism rule.
- [ ] Noise-bucket size is reported as its own measured number, not silently absorbed.
- [ ] Human review: both annotators independently read the same sample of clusters
      (exemplars + random draw) and record a coherence verdict; raw agreement rate is
      reported. Sample size and any target agreement level are open — see below.
- [ ] `docs/specs/typologies/TEMPLATE.yaml` schema is written and validated by a test: a
      config referencing a `cluster_id` absent from its named run's `cluster_summary.jsonl`
      fails validation. The test may use a toy config, not the (not-yet-written) physics one.
- [ ] Unit tests cover: sentence dedup/id construction and its join back to
      `candidates.jsonl`, the embedding→cluster join, determinism under fixed seeds, and
      typology-contract validation.

## Open questions

- **Does `all-MiniLM-L6-v2` + UMAP + HDBSCAN actually produce coherent clusters on this
  corpus?** Unmeasured — the first run and human review answer this; if not, the
  Alternatives table has the fallbacks to try next.
- **Cluster-review sample size and what agreement rate counts as "good enough"** — needs a
  number once the actual cluster count from a first run is known.
- **How many clusters emerge, and what fraction is noise** — unmeasured until the stage
  runs once.
- **Should `cluster_id` be made stable across reruns** (e.g. by canonicalizing on centroid
  or exemplar-set similarity to the previous run) instead of being run-scoped? Left as-is
  for now; revisit if maintaining plug-in mappings across re-clustering proves painful.
- **Annotation budget for the physics-exemplary typology** (the follow-on spec) is still
  unset — it depends on how many clusters emerge and is decided once that list exists.
- **Do `subject_link` and `link_link` candidates cluster differently?** Worth checking once
  clusters exist: `subject_link` is known (spec 0002 Risks) to over-fire the subject
  heuristic; if those candidates land disproportionately in noise or in one diffuse cluster,
  that is direct evidence of the heuristic's error rate.

## References

- Sentence-BERT — Reimers & Gurevych (2019), https://arxiv.org/abs/1908.10084
- UMAP — McInnes, Healy, Melville (2018), https://arxiv.org/abs/1802.03426
- HDBSCAN — Campello, Moulavi, Sander (2013), density-based clustering with a native noise
  label, https://doi.org/10.1007/978-3-642-37456-2_14
- BERTopic — Grootendorst (2022), the embed → reduce → cluster → interpret pattern this
  spec follows, https://arxiv.org/abs/2203.05794
- GLiREL — already referenced in spec 0002; relevant again once a typology plug-in exists.

## Changelog

- 2026-08-26 — created. Scoped as a field-agnostic clustering interface plus a typology
  plug-in contract, so the relation typology can vary by research field without changing
  the clustering stage; the concrete physics label set is deferred to a follow-on spec.
- 2026-08-26 — accepted. Implementation started: `cluster` stage
  (`src/inpnet/relations/cluster.py`), the typology contract and validator
  (`src/inpnet/relations/typology.py`, `docs/specs/typologies/TEMPLATE.yaml`), and
  `inpnet cluster`. Not yet run against the corpus — acceptance criteria that need a
  real run (determinism check, noise-bucket size, human review) are still open.
- 2026-08-26 — first full-corpus run, and a correction to §Decision 2-3. Default
  settings (`cluster_selection_method="eom"`, UMAP's Euclidean metric) collapsed the
  corpus into one 43,634-sentence cluster. Fixing the metric to `"cosine"` alone did not
  fix it (43,639/20 split, still degenerate). Diagnosed with a new dev tool,
  `inpnet cluster-sweep` (`src/inpnet/relations/diagnostics.py`, outside this spec's
  Interface — see its module docstring): embeds a sample once and sweeps UMAP/HDBSCAN
  hyperparameters against it cheaply. The actual fix was `cluster_selection_method`:
  HDBSCAN's default `"eom"` (excess of mass) favors the most *persistent* cluster in the
  condensed tree, which at corpus scale is the root; `"leaf"` selects the tree's leaves
  instead and stayed balanced from a 5,000-sentence sample up through the full corpus.
  Full run (`--cluster-selection-method leaf --n-neighbors 15 --n-components 15
  --min-cluster-size 5`, cosine metric, 4m42s): **1,286 clusters, 27,881 noise (63.8%)**,
  median cluster size 9, 609 clusters ≥10 sentences. Spot-checked exemplars read as
  genuinely relation-like — a doctoral-mentorship cluster ("his doctoral students
  include..."), a postdoc-supervision cluster, a family-relation cluster ("grandson of,"
  "great-nephews were," "brother of"), and named research collaborations (the DNA
  structure discovery; black-hole physics) — alongside some clusters that read as shared
  historical/topical narrative rather than one specific relation, which the human-review
  step (§Decision 5) still needs to sort out. `cluster_selection_method` added as a
  first-class, swept parameter next to `metric`; `"eom"` kept as the library default but
  no longer this project's default. The two annotators' formal coherence review and the
  determinism/byte-identical rerun check are still open.
- 2026-09-28 — `cluster`'s output directory moved from the flat `data/interim/` to
  `data/interim/relation_typology/{version}/` (starting at `0.0.1`), reading spec 0002's
  `candidates.jsonl` from its own versioned directory. Small correction to this Accepted
  spec's documented CLI interface, not a decision reversal — see `data_versions.json` and
  AGENTS.md §4.
- 2026-09-28 — added §Decision 2a, entity masking before embedding, prompted by a
  concrete instance of the bias `docs/findings/0001-relation-clustering-entity-and-template-bias.md`
  Observation 3 already flagged: three sentences about Henry B. Eyring asserting three
  different relations clustered on his name alone. `mentions.jsonl`/`entities.jsonl`
  added as `cluster` stage inputs; implemented in `src/inpnet/relations/cluster.py`
  (`mask_sentence`, `build_entity_index`, `apply_masking`) and threaded through
  `inpnet cluster` (new `--mentions`/`--entities`/`--no-mask-entities` flags) and
  `inpnet cluster-sweep`. `data_versions.json`'s `relation_typology` bumped to `0.0.2` —
  masking changes what's embedded, so it gets its own version directory rather than
  overwriting the `0.0.1` run.
- 2026-09-28 — `0.0.2` full-corpus run (same hyperparameters as the `0.0.1` run:
  `cluster_selection_method=leaf`, `n_neighbors=15`, `n_components=15`,
  `min_cluster_size=5`; `clustering_run_id 28755104d83f`). All 43,664 sentences had at
  least one detected person mention masked. **1,224 clusters, 28,862 noise (66.1%)** —
  close to but not identical to the unmasked run (1,286 clusters, 63.8% noise); masking
  measurably perturbs cluster structure without being a dramatic reshuffle. Direct check
  against this spec's own motivating example: the three Henry B. Eyring sentences quoted
  in §Decision 2a landed in the same cluster in both runs (cluster 10 unmasked, cluster
  57 masked) — masking did not split them apart. What did change: unmasked, that cluster
  is exactly the 7 sentences from Eyring's own article and nothing else; masked, it
  gains one sentence from a different article (Q61951207, "Meserve served as a law
  clerk to Justice Benjamin Kaplan... and Justice Harry Blackmun...") — the same "served
  as [role] to/of [institution]" career-summary template, different person. Reads as
  masking nudging the cluster from purely entity-driven toward genuinely
  template/relation-driven, but only partially: these three sentences may still cluster
  because they share that institutional-office template with each other, not because
  they share "Eyring" — masking removes the name but not the template, and this run
  doesn't distinguish the two. Recorded as an honest partial result, not a fix claim;
  see
  [`docs/findings/0002-entity-masking-effect-on-relation-clustering.md`](../findings/0002-entity-masking-effect-on-relation-clustering.md).
  Determinism (byte-identical rerun) and the two-annotator coherence review are still
  open for this run, same as `0.0.1`.
- 2026-09-28 — **bug found and fixed in link-mention masking; corrected `0.0.2` run.**
  A user-reported check against real Wikipedia sentences ("Sir Humphry Davy" appearing
  in three unrelated articles' sentences, still clustering together after masking)
  traced to `build_entity_index` slicing a link mention's `start`/`end` directly against
  the sentence-local text. Per spec 0002 §Decision 4, those offsets are
  **section-relative**, not sentence-relative — e.g. one real mention span was
  `[358, 374]` against a sentence only 124 characters long. Python slicing doesn't raise
  on an out-of-range index, it clips, so `text[:358]` silently returned the whole
  sentence unchanged and `text[374:]` returned empty — the placeholder was appended at
  the very end instead of removing anything, and the name was left fully intact. This
  affected every *linked*-person mask (everyone except the article's own subject, which
  is matched by name/regex, not offset, and was unaffected) for any sentence that isn't
  the first in its section — effectively most sentences in the corpus. Explains why the
  `0.0.2` run above still showed the Eyring sentences clustering together: the "fix" it
  measured wasn't actually removing linked names at all, only the subject.

  Fixed by adding `build_sentence_starts` (reads each sentence's own section-relative
  start from `sentences.jsonl`, spec 0002's output) and having `build_entity_index`
  convert every link span to sentence-local before storing it; `mask_sentence` also
  gained a defensive bounds check that drops any span still outside the text after
  conversion rather than silently corrupting it. `sentences.jsonl` is now a fourth input
  to the `cluster` stage (`--sentences`, defaulting like the others). New regression
  tests in `tests/test_cluster.py` encode section-relative offsets with a nonzero
  section-start explicitly, so a reversion to sentence-relative slicing fails loudly
  instead of passing by coincidence, the way the original tests did.

  **Corrected full-corpus run** (same hyperparameters, same `clustering_run_id
  28755104d83f` — the run id is derived from config, not code, so an unrelated code fix
  doesn't change it even though the output does; a real limitation of content-hashing
  only the config, noted here rather than fixed, since nothing outside this session has
  referenced that run id yet): **1,232 clusters, 28,542 noise (65.4%)**; 43,648 of
  43,664 sentences (99.96%) had at least one mention actually masked (down from a
  vacuous 100% under the bug, where every sentence counted as "masked" even when nothing
  was removed). Direct re-check of the motivating examples: **all three Eyring sentences
  are now noise (`cluster_id -1`) — no longer clustered together at all** — and so are
  all three "Sir Humphry Davy" sentences from the bug report. Overwrote the buggy
  `0.0.2` output in place (same version — a bug fix to what `0.0.2` was always supposed
  to mean, not a new config decision); see
  [`docs/findings/0002-entity-masking-effect-on-relation-clustering.md`](../findings/0002-entity-masking-effect-on-relation-clustering.md)'s
  addendum for the full before/after comparison.
- 2026-09-28 — **superseded.** A user review of the corrected `0.0.2` run surfaced the
  same bias one level up: with person names masked, clusters still cohered on shared
  *non-person* context — a shared place ("Harwell"), shared topic ("crystal growth"),
  or both — rather than on relation type (e.g. the BCF-theory / crystal-growth cluster
  and the AERE-Harwell cluster quoted during discussion). Masking more entity types
  would have been a direct extension of §2a, but the discussion that followed settled on
  a bigger change instead: don't discover relation structure via one whole-sentence
  embedding at all — pull out each sentence's individual factors (time, place,
  institution, the actual relation-bearing action) as separately detected, typed spans,
  and represent a relation as an open, extensible bag of those attributes rather than a
  single `cluster_id`. That doesn't fit this spec's interface (one `cluster_id` per
  sentence) or its explicitly field-agnostic-clustering framing, so it's a reversal, not
  an amendment — replaced by spec 0004 (attribute span detection) and spec 0005
  (the schemaless relation record), per `docs/specs/README.md`'s "reversing a decision"
  rule. This spec's code and findings are not deleted; `src/inpnet/relations/cluster.py`
  and its masking logic (§2a) remain available as a baseline/ablation, and spec 0004
  reuses its sentence-local span-conversion logic directly.
