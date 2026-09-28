# 0001 — Corpus acquisition

| | |
|---|---|
| **Status** | Draft |
| **Depends on** | none |
| **Superseded by** | — |
| **Owner** | Falk Minks |
| **Last updated** | 2026-08-12 |

## Context

Everything downstream inherits this corpus's biases and its noise, so this is the first
decision that has to be recorded.

The project extracts interpersonal networks between **real people**, which puts two
demands on the corpus that a general text collection would not have. First, it must
contain only actual humans — a fictional character or a film in the node set is not a
mild annoyance, it is a correctness failure in a graph asserting who knew whom. Second,
it should preserve as much identity signal as possible, because linking a mention to a
person is a hard problem we would rather not solve from scratch.

The first pass targets **physicists on English Wikipedia**: a densely interconnected
community with thorough article coverage.

## Goal

A reproducible, revision-pinned set of English Wikipedia biographies of physicists,
stored as cleaned prose with wiki-link spans preserved, alongside the Wikidata metadata
needed to identify each subject.

## Non-goals

- Languages other than English (first pass).
- Sources other than Wikipedia.
- Full-dump processing of all of Wikipedia.
- Extracting infoboxes as structured data — they largely duplicate Wikidata, which the
  seed stage already retrieves. We want running prose.
- Deciding the final corpus *size*. This spec produces the full seed list; how much
  article text gets fetched in the first pass is settled once the size is measured.

## Decision

### 1. Selection — Wikidata SPARQL, not the category tree

The seed set is defined by [`docs/queries/physicists.rq`](../queries/physicists.rq):

```sparql
?person wdt:P31 wd:Q5 ;                    # instance of: human
        wdt:P106/wdt:P279* wd:Q169470 .    # occupation: physicist (incl. subclasses)
?article schema:about ?person ;
         schema:isPartOf <https://en.wikipedia.org/> .
```

**Why not the Wikipedia category tree.** `Category:Physicists` has 28 direct
subcategories. Among them: `Fictional physicists`, `Films about physicists`,
`Cultural depictions of physicists`, `Lists of physicists`,
`Lists of things named after physicists`, `Physicist stubs`, and
`Wikipedia categories named after physicists`.

Recursing that tree therefore admits films, fictional characters, list articles and
maintenance categories directly into the corpus. Beyond contamination, the tree has no
safe depth limit (`Physicists by nationality` → country categories → unrelated drift) and
the category graph contains cycles. Excluding all of this by category blocklist means
maintaining a blocklist forever; `P31 = Q5` excludes it structurally.

**Two constraints discovered while implementing this** (2026-08-12), both now
encoded in the query file and the seed stage:

- The query must **not** use `SERVICE wikibase:label`. With it, the full query runs 67s
  and is truncated by the WDQS timeout; without it, 9.6s and a clean result. Display
  names are taken from the article title instead, which is the same string a reader sees.
- **WDQS signals that timeout as HTTP 200.** It streams results and appends a Java stack
  trace to the partial JSON. A client that only checks the status code silently accepts an
  incomplete corpus, so `WikiClient.sparql` detects the truncation markers and raises.

Selecting via Wikidata also yields, for free:

- a **QID per person** — the identifier the entity-linking stage will need anyway,
- birth/death year, gender, and occupations — the metadata a coverage-bias analysis
  requires,
- a selection criterion that is **one version-controlled query string**, citable in the
  paper rather than described in prose.

The cost is real and should be stated in the paper: people whom English Wikipedia
categorises as physicists but whose Wikidata item lacks `P106` are silently missed. See
*Open questions* on measuring that gap.

### 2. Retrieval — live REST API, revision-pinned

Fetch each seed article from the Wikimedia REST API and **record its revision id**. The
live API returns whatever an article says today, so without pinning a rerun months later
silently produces different numbers — unacceptable for a paper. Pinning makes the
snapshot citable and the analysis repeatable.

Verified endpoints:

| Purpose | Endpoint |
|---|---|
| Article HTML **+ revision id** | `GET /api/rest_v1/page/html/{title}` → Parsoid HTML; response `ETag` is `W/"{revision_id}/{uuid}/view/html"` |
| Metadata only | `GET /w/rest.php/v1/page/{title}/bare` → `latest.id`, `latest.timestamp`, `license`, `html_url` |
| Permanent link | `https://en.wikipedia.org/w/index.php?oldid={revision_id}` |

**One request per article.** The HTML response's `ETag` already carries the revision id
(verified: `Niels_Bohr` → `W/"1366956682/…"`), so the pin comes free with the content and
the `/bare` call is not needed on the happy path. Keep `/bare` for pre-flight checks —
resolving redirects or confirming a title exists — not for every fetch.

At the measured corpus size (below), one polite request per second puts a full fetch at
roughly 4–5 hours of wall-clock. That is an overnight job, not a reason to switch to the
dump.

**Etiquette is a requirement, not an optimisation.** Wikipedia explicitly discourages
bulk crawling. The fetcher sends a descriptive `User-Agent` with a contact address,
requests serially with a delay, retries with backoff, and never runs unthrottled.

Raw responses are written to `data/raw/` **unmodified**, so cleaning can be re-run and
iterated without re-fetching a single article.

The XML dump (`pages-articles-multistream`, >25 GB compressed / ~105 GB expanded, random
access per ~100-page stream via its index) remains the documented fallback if the seed
list turns out large enough that polite fetching becomes impractical.

### 3. Format — Parsoid HTML, cleaned to prose with link spans

Store Parsoid HTML, then clean it to prose.

The reason is **wiki links**. Parsoid emits them as `<a rel="mw:WikiLink" href="./Page_Title">`,
which is both machine-extractable and *human-curated*: a link to `./Niels_Bohr` is an
entity link a person wrote deliberately. That is the single most valuable identity signal
in the corpus, and plain-text extracts discard it. Parsoid also resolves templates, which
raw wikitext would leave us to interpret.

**Cleaning rules.** Keep: prose paragraphs, and section headings (section context such as
"Later life" or "Controversy" is a cheap and useful feature downstream). Drop: infoboxes,
navboxes, tables, reference lists and footnote markers, `See also` / `External links` /
`Further reading` / `Notes` sections, image captions, edit links, and math markup.

Wiki links are preserved as **character offsets into the cleaned text**, not as inline
markup, so downstream stages can treat the text as plain prose and still recover the links.

### 4. Measured corpus size

Run 2026-08-12 against the Wikidata Query Service, humans with an English Wikipedia
article:

| Selection | Count |
|---|---|
| `P106 = Q169470` (strict) | **11,971** |
| `P106/P279* → Q169470` (incl. subclasses) | **15,158** |

Subclass traversal adds **3,187 people (+26.6%)** — astrophysicists, theoretical
physicists, and similar. Both numbers are small enough that the full corpus is
tractable; the choice between them is a scope question, not a feasibility one.

**Projected download size** (measured 2026-08-12 by `inpnet estimate` over the full
subclass seed list). The wikitext total is *measured, not estimated* — the Action API's
`prop=info` returns each page's exact wikitext byte count, 50 titles per request:

| | |
|---|---|
| Articles resolved | 15,157 (0 missing, 27 redirects) |
| Wikitext total | **151.6 MB** (exact) |
| Wikitext mean / median / max | 10.2 KB / 7.1 KB / 269 KB |
| HTML : wikitext ratio | **7.87×** (size-weighted, sampled on 40 articles) |
| **Projected HTML** | **~1.2 GB** |

Sampling for that ratio must be drawn from the real seed list. Famous physicists
(Einstein, Curie) average ~120 KB of wikitext against the corpus mean of 10.2 KB, so a
sample of well-known names overshoots the projection by roughly 10×.

### 5. Scope — full seed list, bounded text fetch first

The `seed` stage runs over the complete result set: it is one query and costs nothing.

The `fetch` stage runs on a **bounded subset first**, so the cleaning rules can be
validated before committing to a long crawl. Scale up once cleaning is verified. Subset
size and selection method are open (below) and will be decided against the measured
corpus size.

## Interface

Three separate commands. A failed fetch must never force a re-crawl, and a cleaning bug
must never force a re-fetch — that separation is the whole reason these are not one script.

| Stage | Reads | Writes |
|---|---|---|
| `seed` | `docs/queries/physicists.rq` | `data/raw/seed.jsonl` |
| `fetch` | `data/raw/seed.jsonl` | `data/raw/html/{qid}.html`, `data/raw/fetch_log.jsonl` |
| `clean` | `data/raw/html/` | `data/interim/corpus_acquisition/{version}/documents.jsonl` |

`{version}` is this layer's current version, recorded in
[`data_versions.json`](../../data_versions.json) at the repo root and bumped by hand —
see AGENTS.md §4.

**`seed.jsonl`** — one object per person:

```jsonc
{
  "qid": "string — Wikidata QID, the stable id used everywhere downstream",
  "title": "string — English Wikipedia article title, also used as display name",
  "birth_year": "int|null",
  "death_year": "int|null",
  "gender": "string|null — QID; kept for the coverage-bias analysis"
}
```

No `occupations` field: aggregating it with `GROUP_CONCAT` pushes the query past the
WDQS timeout. It is recoverable later per-QID if a stage actually needs it, and nothing
currently does — the bias analysis needs gender and birth year, which are present.

**Deduplication is mandatory, not defensive.** The `OPTIONAL` clauses emit a row per
combination, so the query returns **18,178 rows for 15,158 distinct people**; 2,625
people carry more than one row because Wikidata holds several birth dates or genders for
them. The seed stage keeps the first value per field, fills gaps from later rows, and
sorts by QID so output is byte-stable across runs.

**`fetch_log.jsonl`** — one object per attempt, so failures are data rather than lost:

```jsonc
{
  "qid": "string",
  "title": "string",
  "revision_id": "int|null",
  "status": "enum — ok | not_found | redirect | error",
  "redirect_to": "string|null",
  "http_status": "int",
  "fetched_at": "string — ISO 8601 UTC",
  "error": "string|null"
}
```

**`documents.jsonl`** — one object per article:

```jsonc
{
  "doc_id": "string — the QID",
  "title": "string",
  "qid": "string",
  "revision_id": "int — pinned revision",
  "url": "string — permanent link including oldid",
  "fetched_at": "string — ISO 8601 UTC",
  "license": "CC BY-SA 4.0",
  "sections": [
    {
      "heading": "string — '' for the lead section",
      "text": "string — cleaned prose, paragraphs separated by \\n\\n",
      "links": [
        {
          "start": "int — char offset into this section's text",
          "end": "int",
          "surface": "string — anchor text as it appears",
          "target_title": "string — linked article title"
        }
      ]
    }
  ]
}
```

`target_title` is deliberately **not** resolved to a QID here — that is the entity-linking
stage's job and belongs in its own spec.

**Manifests.** Every stage writes `_manifest.json` next to its output recording: input
file hashes, the config used, tool and model versions, timestamp, and output counts. For
`seed` this includes the query text and the query date.

## Alternatives considered

| Option | Why not (for now) |
|--------|-------------------|
| Wikipedia category tree recursion | Admits fictional characters, films, and list articles; no safe depth; cyclic graph. See §Decision 1. |
| XML multistream dump | Fully reproducible and rate-limit-free, but 25 GB plus wikitext parsing is disproportionate when the seed list is a bounded set of titles. Documented as the fallback. |
| Wikimedia Enterprise HTML dumps | The free public mirror on `dumps.wikimedia.org` was **discontinued in March 2025**; access now needs an Enterprise account and API token. Noted so this isn't reconsidered from stale knowledge. |
| HuggingFace `wikimedia/wikipedia` | Convenient, but wiki links are stripped and the cleaning is opaque — and the links are exactly what we want. |
| Action API `prop=extracts` (plain text) | Simplest, but discards wiki links. |
| Raw wikitext | Smallest, links present as `[[...]]`, but template resolution is a project of its own. |
| Live API without revision pinning | Not reproducible; the corpus would drift under the analysis. |

## Risks

- **Domain bias.** Physicist coverage on Wikipedia skews male, Western, and modern. Every
  downstream network metric inherits that skew. This must be quantified using the seed
  metadata and stated in the paper — not left for a reviewer to notice.
- **Fetch runtime**: ~15k articles at one polite request per second is an overnight run.
  Mitigated by caching raw HTML so it is paid once, and by validating cleaning on a subset
  before committing to the full crawl.
- **Article length variance** is extreme, from stub to Einstein. Report the distribution;
  it will matter when aggregating evidence later.
- **Parsoid HTML structure drift** could break the cleaner. Mitigated by keeping raw HTML
  on disk so cleaning can be re-run without re-fetching.
- **Occupation metadata is incomplete** in Wikidata, so the seed set under-covers by an
  unknown amount. See *Open questions*.

## Acceptance criteria

- [x] Corpus size measured with [`physicists-count.rq`](../queries/physicists-count.rq);
      both counts recorded in §Decision 4 — 11,971 strict / 15,158 with subclasses.
- [x] `seed.jsonl` produced for the full result set (15,158 people), matching the schema above.
- [x] Every successfully fetched document carries a `revision_id` and a permanent `oldid` URL.
- [x] Re-running `clean` on an unchanged raw snapshot produces byte-identical output
      (verified by SHA-256 over two consecutive runs).
- [x] Link offsets round-trip **and** carry no surrounding whitespace — the second check
      exists because round-tripping alone passed while offsets and surface were both
      wrong by one character. See `verify_offsets` and its regression tests.
- [x] Fetcher sends a contact `User-Agent`, rate-limits, and retries with backoff
      honouring `Retry-After`.
- [x] Manifests written for all three stages, including the seed query and its date.
- [x] Cleaning hand-verified across the length distribution (stub of 59 chars through
      von Neumann at 92k), plus a full-corpus artefact scan: **0** occurrences of
      `[edit]`, `Archived from`, `^ a b`, or `.mw-parser` in 53.1 M chars. The 105 `ISBN`
      and 40 `doi:` hits were inspected and are legitimate prose ("Published in 2009 by
      Penguin (ISBN …)"), not reference-list leakage.
- [x] Full-corpus fetch complete: 15,158 articles, 0 errors, 1.2 GB — matching the
      1.2 GB projection.
- [ ] Failed fetches appear in `fetch_log.jsonl` — **done for errors, not for redirects.**
      The REST HTML endpoint follows redirects transparently, so `fetch` cannot see them;
      `estimate` detects them via `prop=info` (27 in the corpus). See *Open questions*.

## Open questions

- **Strict `P106` or `P279*` subclass traversal?** Measured: 11,971 vs 15,158 (§Decision 4).
  `physicists.rq` currently uses the subclass form. Confirm that is wanted — the extra
  3,187 are astrophysicists and similar, which arguably belong, but strict `P106` is the
  tighter definition and easier to describe in the paper. **Needs a human decision.**
- **First-pass subset**: how large, and selected how — random sample, by era, by article
  length, or by link degree within the seed set? A random sample is the defensible default
  unless there is a reason to stratify. Note the full corpus is now known to be tractable,
  so the subset is about iterating quickly, not about affordability.
- **Redirects**: 27 of 15,157 seed titles redirect. Wikidata sitelinks are almost always
  canonical, so this is a small effect — but the REST HTML endpoint follows redirects
  transparently, so `fetch` records the *requested* title while the content is the
  target's. Options: pre-resolve via `prop=info` during `seed`, or accept it and note it.
  Low stakes, but currently silent, which is the part worth fixing.
- **Minimum article length** to exclude stubs — or keep stubs and let downstream stages
  ignore them?
- **Category-tree coverage comparison**: worth fetching category membership separately to
  measure how many category-listed physicists Wikidata's `P106` misses? Would produce a
  reportable number about the selection method at modest cost.

## References

- Seed query: [`docs/queries/physicists.rq`](../queries/physicists.rq)
- Size measurement: [`docs/queries/physicists-count.rq`](../queries/physicists-count.rq)
- Wikidata Query Service — https://query.wikidata.org/
- MediaWiki REST API — https://www.mediawiki.org/wiki/API:REST_API
- Parsoid output spec (`rel="mw:WikiLink"`) — https://www.mediawiki.org/wiki/Parsoid
- Wikipedia:Database download — https://en.wikipedia.org/wiki/Wikipedia:Database_download
- Wikimedia Enterprise HTML dumps — https://dumps.wikimedia.org/other/enterprise_html/

## Changelog

- 2026-09-28 — **fixed: HTML comments were leaking into cleaned prose.** A spec 0004
  diagnostic found sentences truncated mid-word at a literal `[`
  (`Q3760460:3:4:Q3760460:Q1922193`: "...formed by Congressman Mervyn ["). Root cause:
  `bs4.Comment` is a `NavigableString` subclass, so `_walk`'s
  `isinstance(child, NavigableString)` check could not tell an HTML comment from real
  text — editors' `<!-- Deleted image removed: [[File:...]] -->` notes and large
  commented-out draft sections (raw, unrendered wikitext, sometimes spanning what look
  like several paragraphs) were appended straight into the extracted prose. Confirmed
  against the raw HTML of `Q283201` and `Q504303`. Scanned the full raw corpus: **1,461
  of 15,158 articles (9.6%) carry HTML comment nodes, ~317K characters total** that were
  leaking in. Fixed by extracting all `Comment` nodes before walking. Re-ran `clean` on
  the full corpus: **15,158 documents, 47,566 sections, 443,527 links, 0 offset
  errors** — identical to the pre-fix counts (§Decision 4's measured numbers still
  hold), confirming no real content or link was lost, only comment garbage. This was
  necessary but not sufficient for the truncation symptom — see spec 0002's Changelog
  for the companion segmentation bug and the combined before/after candidate counts.
  Regression tests: `test_html_comments_are_not_included_as_prose`,
  `test_comment_spanning_multiple_paragraphs_is_removed` in `tests/test_clean.py`.
- 2026-08-12 — created.
- 2026-08-12 — corpus size measured against WDQS: 11,971 strict / 15,158 with subclass
  traversal. Verified the HTML endpoint's `ETag` carries the revision id, so fetching is
  one request per article rather than two.
- 2026-08-12 — implemented as `inpnet seed|estimate|fetch|clean` and run end-to-end.
  Spec revised from what the implementation found:
  - Removed `SERVICE wikibase:label` from the seed query (67s + truncated → 9.6s + clean)
    and dropped `occupations` from `seed.jsonl` (`GROUP_CONCAT` exceeds the WDQS timeout).
  - Recorded that WDQS reports timeouts as HTTP 200 with a truncated stream, and that the
    client must detect it.
  - Recorded that `OPTIONAL` clauses duplicate rows (18,178 → 15,158; 2,625 affected).
  - Added the measured download projection: 151.6 MB wikitext → ~1.2 GB HTML at 7.87×.
  - Strengthened the link-span check after finding spans that round-tripped while being
    shifted one character onto preceding whitespace.
- 2026-08-12 — **full corpus acquired.** Fetch: 15,158 articles, 0 errors, 1.2 GB (the
  1.2 GB projection was accurate). Clean: 15,158 documents, 47,566 sections,
  443,527 wiki links, 0 offset errors under strict mode, 100 MB of `documents.jsonl` —
  53.1 M characters, ~10.6 M words of prose.
- 2026-09-28 — `clean`'s output moved from the flat `data/interim/documents.jsonl` to a
  per-layer, hand-versioned `data/interim/corpus_acquisition/{version}/documents.jsonl`
  (starting at `0.0.1`), so a rerun doesn't silently overwrite an existing snapshot and so
  this stage's `_manifest.json` stops colliding with 0002's and 0003's, which used to share
  the same flat directory. See `data_versions.json` and AGENTS.md §4.
