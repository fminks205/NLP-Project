# 0006 — Graph visualization

| | |
|---|---|
| **Status** | Draft |
| **Depends on** | 0002, 0005 |
| **Superseded by** | — |
| **Owner** | Falk Minks |
| **Last updated** | 2026-09-28 |

## Context

Every prior spec produces data — mentions, candidates, relation records — but nothing yet
turns any of it into a picture. For both the paper and for sanity-checking the pipeline
by eye, a human needs to actually *look at* a piece of the graph: nodes are people, edges
are the relations spec 0005 assembled, each carrying its own bag of `time`/`place`/
`institution`/`action` attributes.

Graphviz is already installed on this machine (`C:\Program Files\Graphviz\bin`, v16.0.0 —
not on `PATH`). A quick feasibility check against the current full-corpus run
(`data/interim/relations/0.0.2/`) found that a filtered subgraph is comfortably within
Graphviz's legible range: relations whose `institution` attribute contains "yale"
(case-insensitive) — **144 of 51,148 `relations.jsonl` records, 198 people; plus 26 of
25,057 `low_coverage_relations.jsonl` records, 43 people** — total on the order of 150–170
edges and 200–240 people. The full corpus (32,546 entities, 51,148+25,057 relation
records) is not: force-directed layout at that density renders as an unreadable hairball
regardless of tool, a limitation of graph drawing at this scale, not of Graphviz
specifically.

This spec covers the first visualization only: a legible subgraph filtered by an
`institution` substring, annotated edges. It is written so a second, third, ... visualization
(an ego-network, a full-corpus overview, a time-sliced view) can be added later without
redoing this one — see Non-goals and Open questions — but only this one is decided here.

## Goal

Given an institution-name substring, produce a Graphviz-rendered image (SVG, with hover
tooltips carrying each edge's info bag) of the people connected by a relation mentioning
that institution, plus the `.dot` source that produced it.

## Non-goals

- **Other visualization types** (full-corpus overview, ego-network around one person,
  time-sliced or animated views, a web-based interactive viewer). This spec's Interface is
  scoped to the institution-filtered case; a general "visualization" abstraction is not
  designed here — see Open questions.
- **Entity-resolving the institution filter.** Matching is a case-insensitive substring
  test against `attributes[].value` where `attr_type == "institution"` — the same NER
  output spec 0004 already produced, not a new resolution step. Known consequence: `"Yale
  University Press"` (a publisher mention) matches alongside `"Yale University"` (an
  affiliation) for a `"yale"` filter. Documented, not fixed, here.
- **Filtering by attribute types other than `institution`**, or combining multiple filter
  predicates. The CLI takes one substring against one attribute type. Generalizing the
  filter is deferred until a second use case actually needs it.
- **Node/edge styling beyond what's specified below** (community detection, layout by
  cluster, sizing by centrality). A legible first pass, not a finished figure. (Per-edge
  attribute-category coloring was later brought *into* scope — see §6 and the Changelog;
  what's still excluded is styling driven by anything beyond the attribute bag itself.)
- **Search or filtering inside the browser** (re-querying by a different institution
  substring, hiding nodes, etc. without re-running `render-graph`). Pan/zoom, originally
  excluded here too, was added after the first review — see §7/Changelog; this line now
  only excludes the *query*-side interactivity, not view navigation.

## Decision

### 1. Data source: both `relations.jsonl` and `low_coverage_relations.jsonl`

A low-coverage record (spec 0005) was split out because its *overall* token coverage was
thin, most often a missing `action` — not because its `institution` attribute is any less
real. Excluding it would silently drop legitimate Yale affiliations (26 of the 170
relations in the feasibility check came from this file) for a reason unrelated to what
this visualization filters on. Both files are read by default; `--high-coverage-only`
restricts to `relations.jsonl` alone for a user who wants only fully-covered records.

### 2. Filter: case-insensitive substring on `institution` attribute values

```
inpnet render-graph --institution-contains yale --out data/processed/graphs/yale/
```

A relation record matches if any of its `attributes` has `attr_type == "institution"` and
`value` contains the given substring, case-insensitively. Matching records' `participants`
(both `qid`s) define the node set; a relation with a `null` participant `qid` (schema
allows it, spec 0005) is skipped and counted, not silently dropped from the count.

### 3. Nodes: one per distinct participant `qid`, labeled from `entities.jsonl`

`label` = `canonical_name`. Every node has degree ≥ 1 by construction (it only exists
because it participated in a matched relation), so no isolate-pruning step is needed.

### 4. Edges: one per unordered person-pair, collapsing multiple matching relations

Two people can be connected by more than one matched relation (different sentences, same
or different documents). These collapse into **one edge per pair**, not one edge per
relation record: a reader expects one line per relationship, and Graphviz's default
handling of true parallel edges (overlapping, not offset, under `neato`) would make the
multi-relation case actively harder to read, not more informative. Full provenance is not
lost — every contributing relation's sentence and attribute bag goes into that edge's
tooltip (§6). See Alternatives considered for the one-edge-per-relation option this
rejects.

### 5. Layout: `neato`, shelling out to the installed binary

`neato` (spring-model layout) suits the ~150–250 node scale measured in the feasibility
check; `dot` (hierarchical) is for DAGs, not undirected social graphs. The Graphviz
`bin/` directory is not required to be importable code — this stage writes a `.dot` file
and calls the `neato` (or user-chosen engine) *executable* via `subprocess`, matching
`WikiClient`'s existing posture of driving external tools rather than reimplementing them.
Binary location: `--graphviz-bin` / `INPNET_GRAPHVIZ_BIN` env var, defaulting to a `PATH`
lookup — config over constants, since this machine's install (`C:\Program
Files\Graphviz\bin`) isn't on `PATH`. `--engine` selects among `neato` (default), `fdp`,
`sfdp` for when a future filter produces a denser subgraph.

`overlap=false` is set on the DOT graph (added after the first render: `neato`'s default
lets nodes sit on top of each other at this density, which is exactly the non-overlapping
placement Graphviz is capable of but doesn't do unless asked). `sep="+12"` gives the
overlap-removal pass a little breathing room; `splines=true` curves edges around nodes
instead of drawing straight through them.

### 6. Edge annotation: short `label` on the graph, full bag in a hover-only tooltip

- `label` (drawn on the graph): the first `action` value among the edge's contributing
  relations, or the matched `institution` value if none has an `action` — **truncated to
  `LABEL_MAX_CHARS` (28) at a word boundary**. This was tightened after the first render:
  spec 0004 measured `action` spans averaging 91 characters, so drawing one directly as a
  label (the original plan) did not actually produce the "compact label" this section
  originally claimed — it produced a sentence fragment cluttering the layout. The
  untruncated text is never lost; see below.
- Full detail (every contributing relation's sentence plus its complete attribute bag) is
  **hover-only**, but not via Graphviz's own `tooltip` attribute as originally planned.
  Graphviz's `tooltip` renders as a bare SVG `<title>`, which browsers show as a slow,
  unstyled OS tooltip that cannot wrap long text sensibly — not good enough once real
  sentences (not the placeholder text this spec was drafted against) were on screen. The
  DOT `tooltip` attribute is still set, as a plain-`graph.svg` fallback for anyone who
  bypasses the HTML wrapper; `graph.html` (§7) strips each edge's `<title>` and replaces it
  with a custom-styled, JS-driven tooltip `<div>` keyed by an explicit `id` set on every
  node/edge in the DOT (`node__{qid}` / `edge__{qid_a}__{qid_b}`).

### 7. Output: `.dot`, `.svg`, and an interactive dark-mode `.html`, under `data/processed/`

Per AGENTS.md's repository layout, `data/processed/` holds "final outputs for the paper" —
a rendered graph is exactly that, not a re-derivable intermediate pipeline layer, so it
does not get its own `data/interim/{layer}/{version}/` treatment. Each run gets its own
named subdirectory (so two filters don't collide):

```
data/processed/graphs/{name}/
  graph.dot
  graph.svg
  graph.html
  _manifest.json
```

`graph.html` (added after the first render, alongside the `overlap`/label/tooltip changes
above) wraps `graph.svg` inline in a page styled dark (`bgcolor`/`fillcolor`/`fontcolor`
are set to a matching dark palette directly in the DOT too, so `graph.svg` opened alone is
already dark, not just the HTML wrapper) and wires the custom tooltip described in §6. This
is the version meant to actually be opened and explored; `graph.dot`/`graph.svg` remain for
anyone who wants the plain Graphviz artifacts (e.g. to re-render at a different size, or
pipe through another Graphviz tool).

`{name}` defaults to a slug of the filter (`yale` for `--institution-contains yale`);
`--name` overrides it. The manifest follows the project's usual shape (AGENTS.md §5):
input file hashes (both relation files' versions), the filter config, node/edge counts,
tool version (`neato -V` output), timestamp.

### 8. Per-attribute-category color-coded boxes, and zoom/pan (added on review)

A single short text label per edge (§6, first version) turned out to read as "some
attribute, category unstated" once real edges were on screen — the reviewer's ask was to
show each present attribute *type* as its own small box, color-coded by category
(`action` yellow, `time` blue, `institution` green, `place` violet — `ATTR_COLORS` in
code), so the category is visible without reading the text or hovering.

Built with Graphviz's own **HTML-like labels** (`label=<<TABLE>...</TABLE>>`, a DOT
feature, not a post-render SVG hack): one `<TD BGCOLOR=...>` per attribute type present
on the edge's *primary* (first) contributing relation, same "primary relation, full
detail in the tooltip" split §6 already used for the single-label version. Multiple
relations collapsed onto one edge (§4) still show only the primary's boxes on the graph;
every contributing relation's full bag remains in the hover tooltip — showing every
collapsed relation's boxes was considered and rejected as clutter at this graph's edge
density (up to several relations per pair). A fixed `viz-legend` overlay in `graph.html`
(and only there — `graph.svg` alone has no legend, just the colors) spells out what each
color means.

Box text uses a shorter truncation (`BOX_LABEL_MAX_CHARS = 20`) than the original
single-label's 28, since several boxes now sit side by side on one edge.

**Pan/zoom** (mouse wheel to zoom centered on the cursor, drag to pan, double-click to
refit the whole graph) is hand-rolled directly in `graph.html`'s inline `<script>` — a CSS
`transform: translate() scale()` on the embedded `<svg>`, driven by plain event listeners
— rather than a JS pan-zoom library from a CDN. Reason: this project already keeps its
data pipeline reproducible offline (the git-tracked Wikidata cache, spec 0002's
`wikidata_cache.jsonl.gz`, exists specifically so a rerun needs no network); pulling a
zoom library from a CDN would make `graph.html` the one artifact in the pipeline that
silently stops working (or renders subtly differently) without internet access. The
hand-rolled version is self-contained: `graph.html` is a single file with no external
`<script src>`.

### 9. Per-box hover, and dropping the redundant `[attr=value]` tooltip suffix (fixed on review)

§8's first implementation had two real defects, found on review rather than assumed:

**The tooltip repeated what the boxes already showed.** The edge-line tooltip still
appended `[time=..., institution=..., action=...]` after the sentence — exactly the same
information §8's boxes now display, colored, on the graph itself. `_edge_tooltip` is now
sentence text only; a secondary contributing relation's own attribute values (boxes are
primary-only, §8) are still reachable, just by reading its sentence in the (now
multi-line, one per contributing relation) edge tooltip rather than a bracketed dump.

**Every box showed the same tooltip regardless of which one was hovered** (observed:
always the last box's, functionally `action`'s). Root cause, found by inspecting
Graphviz's actual SVG output rather than assumed: an HTML-like-label `<TD>` with a bare
`ID=` and nothing else gets **no SVG element at all** — Graphviz silently drops it. It
only wraps a cell in an addressable element when that cell *also* carries `TOOLTIP=` (or
`HREF=`), and when it does, the id gets an `a_` prefix (`svg_box_id` in code; verified
against this machine's Graphviz 16.0.0, undocumented behaviour). §8's boxes had `ID=` but
no `TOOLTIP=`, so **no box was ever individually present in the DOM** — every hover
inside an edge's area fell through to the one thing that *was* addressable, the edge
itself, whose JS listener always showed the same (last-set) content regardless of cursor
position within it.

Fixed two ways together: (1) every box now also gets `TOOLTIP="{attr}: {value}"` — full,
untruncated, so it also serves as the plain-`graph.svg` native fallback, same role the
edge-level `tooltip` already plays; (2) `graph.html`'s hover logic was rewritten from one
listener per `g.edge` element to a single **delegated** listener on the canvas that calls
`event.target.closest('[id^="a_box__"]')` first and only falls back to
`closest("g.edge[id]")` if no box matched — giving the more specific element priority by
construction rather than relying on listener-attachment order or `mouseenter` bubbling
quirks across nested elements.

## Interface

| Stage | Reads | Writes |
|---|---|---|
| `render-graph` | `data/interim/relations/{version}/relations.jsonl`, `low_coverage_relations.jsonl`; `data/interim/entity_mention_layer/{version}/entities.jsonl` | `data/processed/graphs/{name}/graph.dot`, `graph.svg`, `graph.html`, `_manifest.json` |

**CLI:**
```
inpnet render-graph \
  --institution-contains yale \
  --relations data/interim/relations/0.0.2/ \
  --entities data/interim/entity_mention_layer/0.0.1/entities.jsonl \
  --out data/processed/graphs/ \
  [--name yale] [--high-coverage-only] [--engine neato] [--graphviz-bin <dir>]
```

No new JSONL schema — this stage consumes spec 0002/0005's existing records and produces
`.dot`/`.svg` (and its manifest, following the existing manifest schema) as its own
contract.

## Alternatives considered

| Option | Why not (for now) |
|--------|-------------------|
| One edge per relation record (true multi-edge) | Graphviz's default parallel-edge handling under `neato` overlaps rather than offsets them, actively hurting legibility at exactly the point (a well-documented pair) where the extra edges would matter most. Collapsing keeps "one relationship = one line" and moves the multi-relation detail into the tooltip instead, where it's read on demand rather than always drawn. |
| The `graphviz` PyPI package (DOT-generation + subprocess wrapper) instead of hand-writing `.dot` | Real candidate, not rejected outright — would remove the need to hand-roll DOT string-escaping (node/edge labels can contain quotes, unicode). Left as an implementation-time choice rather than decided here, since it's a thin wrapper either way and doesn't change this spec's data contract. |
| `sfdp` as the default engine | Built for much larger graphs (thousands of nodes); at ~200 nodes `neato`'s layout quality is better and runtime difference is immaterial. `sfdp` stays available via `--engine` for a denser future filter. |
| Entity-resolve the institution filter against Wikidata (P108 "employer", P69 "educated at") instead of substring-matching NER output | More precise (would exclude "Yale University Press"), but a materially bigger effort — new Wikidata queries, a resolution step — for the first visualization. Noted as the natural fix if false positives prove to matter; not attempted here. |

## Risks

- **False positives from substring matching** (`"Yale University Press"` alongside
  `"Yale University"`) are real and currently unfiltered — see Non-goals. Low volume in
  the one filter measured (feasibility check didn't separately count how many of the 170
  matches are false positives; worth doing once this is built).
- **Edge collapsing loses per-relation granularity** in the default view (mitigated by
  putting full detail in the tooltip, but a reader looking at a static PNG export, not the
  SVG, only sees the compact `label`).
- **Institution NER quality** is inherited unvalidated from spec 0004 (its own Risk
  section: `en_core_web_sm` isn't tuned for domain terms) — a mistyped or missed
  `institution` span here means a missed or wrong node in the filtered graph, not
  something this stage can detect.
- **Graphviz not on `PATH`** on this machine specifically — mitigated by the
  `--graphviz-bin`/env var config, but worth confirming the acceptance run actually
  exercises that path rather than a machine where it happens to already be on `PATH`.
- **`graph.html`'s tooltip needs JavaScript.** With JS disabled the page still shows the
  graph (dark, non-overlapping) but hovering shows nothing — `graph.svg`'s native
  `tooltip`-derived `<title>` is the fallback for that case, at the cost of the plain,
  unstyled OS tooltip §6 moved away from. Not treated as a real risk (a research tool run
  locally, not a public page), but worth stating rather than leaving implicit.
- **28-character label truncation is an arbitrary number**, chosen to fit comfortably
  inside a node-sized label at this graph's density, not measured against a real
  legibility test. Easy to retune (`LABEL_MAX_CHARS`) if real use shows it's too
  aggressive or not aggressive enough.

## Acceptance criteria

- [x] `inpnet render-graph --institution-contains yale` runs end to end and produces
      `graph.dot`, `graph.svg`, `graph.html`, `_manifest.json` under
      `data/processed/graphs/yale/`. Run against the full-corpus `relations/0.0.2`:
      170 matched relations, 217 nodes, 162 edges, 0 skipped for a null participant.
- [x] Manifest's node/edge counts match an independent count from `relations.jsonl` +
      `low_coverage_relations.jsonl` (the same check done for this spec's feasibility
      analysis, now automated rather than ad hoc) — 170/217/162 above matches the
      144+26=170 feasibility count.
- [x] Every rendered edge's tooltip contains at least one real sentence excerpt —
      spot-checked by hand against the source `relations.jsonl` records it collapsed, and
      confirmed programmatically in a live browser: all 162 edges resolve a tooltip, e.g.
      "He was awarded the 1995 American Physical Society Edward Bouchet award and the
      2016 Yale University Bouchet Leadership Award Medal. [time=1995,
      institution=American Physical, ...]".
- [x] Non-overlapping placement (`overlap=false`) verified visually on the full 217-node
      run — no node-on-node collisions.
- [x] `graph.html`'s custom tooltip verified in a live browser (not just unit tests):
      dispatching `mouseenter`/`mousemove`/`mouseleave` on a sample edge shows/positions/
      hides the styled tooltip div, and the edge's native `<title>` is confirmed absent
      (stripped) beforehand. The on-graph label for that same edge renders truncated
      ("He was awarded the 1995…") while the tooltip carries the untruncated sentence.
- [x] Unit tests: filter predicate (substring, case-insensitivity, null-`qid` skip), node
      dedup, edge collapsing (two relations between the same pair produce one edge with a
      two-entry tooltip), DOT-escaping of a label containing a quote character, label
      truncation at a word boundary, dark-theme/`overlap=false` DOT attributes present,
      native `<title>` stripped from `graph.html`, tooltip JSON escaped against breaking
      out of its `<script>` tag, HTML-like edge label present with the right `BGCOLOR` per
      attribute type, box values HTML-escaped, boxes drawn from the primary relation only,
      pan/zoom event listeners present with no external `<script src>`, legend lists every
      `ATTR_COLORS` entry, per-box `ID=`+`TOOLTIP=` both present (§9), `svg_box_id`'s
      `a_` prefix, delegated hover checks a box before falling back to its edge, no
      leftover per-element `querySelectorAll` listeners. 40 tests, `tests/test_graph.py`.
- [x] §8's additions verified in a live browser, not just unit tests: legend renders with
      correct swatch colors; scroll-to-zoom and drag-to-pan both confirmed visually on the
      full 217-node run (zoomed in on a cluster, panned to a different region); box text
      and colors match the legend (e.g. a blue `1772` box, a green `Yale University` box,
      a yellow `married the` box on adjacent edges).
- [x] §9's fix verified in a live browser via real DOM hit-testing (`elementFromPoint` +
      dispatched events), not just unit tests: on a 3-box edge, hovering each box in turn
      showed its own distinct tooltip ("Time: 1995", "Institution: American Physical",
      "Action: He was awarded the 1995 American Physical Society Edward Bouchet award" —
      the full, untruncated value), and hovering the bare edge line between boxes showed
      the sentence only, no `[attr=value]` suffix. Before the fix: 0 addressable box
      elements existed in the DOM at all (`totalBoxes: 0`), confirming the root cause.
- [ ] Re-running on unchanged input reproduces a byte-identical `graph.dot` (AGENTS.md §5
      determinism) — not yet explicitly re-verified after this round of changes, though
      nothing in `to_dot` depends on wall-clock time or iteration order. `graph.svg`/
      `graph.html` are allowed to vary only in whatever Graphviz itself doesn't guarantee
      byte-stable (e.g. embedded timestamps).

## Open questions

- **A general visualization interface** (so an ego-network or full-corpus overview reuses
  the node/edge-building code rather than duplicating it) is worth designing once a second
  visualization type is actually needed — not before, per this project's "don't design for
  hypothetical future requirements" convention. Candidate shape: split "select relations →
  build node/edge lists" from "render node/edge lists to DOT," so only the first half
  changes per visualization type.
- **Should false-positive institution matches (`"... Press"`, `"... Foundation"` etc.) be
  suppressed by a small denylist**, or left for the Wikidata-resolved version (see
  Alternatives)? Not decided — depends on how much noise the first real run shows.
- **PNG export** alongside SVG — trivial to add (`neato -Tpng`) if the paper ends up
  wanting a raster figure; not included in v1 since SVG's tooltips are the more useful
  default for exploration.

## References

- Graphviz documentation — https://graphviz.org/documentation/
- `neato` layout — https://graphviz.org/docs/layouts/neato/
- Spec 0005's relation record schema (this spec's primary input).

## Changelog

- 2026-09-28 — created. Feasibility measured against the full-corpus `0.0.2` relations run
  before writing this spec, not assumed: 144 `relations.jsonl` + 26
  `low_coverage_relations.jsonl` records match `institution` containing "yale", 198 + 43
  distinct people (some overlap between the two files, not yet deduplicated in this
  count).
- 2026-09-28 — implemented (`src/inpnet/viz/graph.py`, `inpnet render-graph`) and run
  against the full corpus: 170 matched relations -> 217 nodes, 162 edges. Three
  corrections made after seeing the first real render, not anticipated when this spec was
  drafted:
  - **Node overlap.** `neato` does not do non-overlapping placement unless told to; added
    `overlap=false` (`sep`/`splines` alongside it). See §5.
  - **The on-graph `label` was not actually compact.** `action` values average 91
    characters (spec 0004); the original plan to draw one directly as a label produced
    clutter, not the "compact label" §6 claimed. Added word-boundary truncation
    (`LABEL_MAX_CHARS = 28`) and moved full detail exclusively into a hover-only view.
  - **Graphviz's own `tooltip` (a bare SVG `<title>`) wasn't good enough** once real
    sentences, not placeholder text, were on screen — slow, unstyled, no line-wrapping.
    Added `graph.html`: the SVG inline in a dark-mode page, native edge `<title>`s
    stripped and replaced with a custom, JS-driven tooltip div. `graph.svg`'s native
    `tooltip` is kept as the fallback for anyone opening it directly. See §6/§7.
  Also: dark color palette (`bgcolor`/`fillcolor`/`fontcolor`) set directly in the DOT, so
  `graph.svg` is dark on its own, not only inside the HTML wrapper. 28 unit tests added;
  the interactive tooltip (title-stripping, show/position/hide, label-vs-tooltip content
  split) additionally verified in a live browser via dispatched mouse events, not just
  unit-tested. Acceptance criteria updated to reflect what was actually run and checked.
- 2026-09-28 — added per-attribute-category color-coded edge boxes and pan/zoom, on
  review — both reverse a Non-goal this spec originally stated (styling beyond §6, and
  interactive pan/zoom); see §8 for the reasoning and the updated Non-goals section.
  `action`/`time`/`institution`/`place` each render as a small colored box (Graphviz
  HTML-like labels, not a post-render hack) built from the edge's primary contributing
  relation; `graph.html` gained a color legend and hand-rolled (no CDN dependency)
  wheel-zoom/drag-pan/double-click-reset. 7 new unit tests (35 total); zoom, pan, and box
  colors all additionally verified in a live browser, not just unit-tested. Acceptance
  criteria updated.
- 2026-09-28 — fixed two defects in §8's boxes, found on review: (1) the edge tooltip
  still appended `[attr=value, ...]` after the sentence, duplicating exactly what the
  boxes now show — `_edge_tooltip` is sentence text only now. (2) every box showed the
  same (effectively the last/`action`) tooltip no matter which was hovered — root cause,
  found by inspecting Graphviz's actual SVG output: a bare `ID=` on an HTML-like-label
  `<TD>` produces **no SVG element at all** unless the cell also has `TOOLTIP=`/`HREF=`,
  in which case Graphviz wraps it in an `a_`-prefixed anchor (`svg_box_id`). No box was
  ever individually present in the DOM; every hover fell through to the one addressable
  thing in the group, the edge itself. Fixed by adding `TOOLTIP=` to every box (also
  serves as `graph.svg`'s native per-box fallback) and rewriting the hover JS from
  one-listener-per-edge to a single delegated listener that checks
  `closest('[id^="a_box__"]')` before falling back to `closest("g.edge[id]")`. 5 new
  unit tests (40 total); the fix verified live via real DOM hit-testing
  (`elementFromPoint` + dispatched events) confirming each of 3 boxes on one edge shows
  its own distinct, full-value tooltip. See §9.
