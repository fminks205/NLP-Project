"""Institution-filtered relation graph -> Graphviz. Implements spec 0006.

Renders a legible subgraph, not the full corpus: nodes are the people who share a
relation whose `institution` attribute contains a given substring; edges collapse
every matching relation between the same pair into one line, annotated from spec
0005's attribute bag. See spec 0006 for why the graph is scoped this way.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..manifest import read_jsonl, write_manifest

DEFAULT_ENGINE = "neato"

#: Characters kept in the on-graph edge label before an ellipsis — spec 0004 measured
#: `action` spans averaging 91 characters, nearly half a typical sentence, so drawing
#: one directly on the graph (the original v1 behaviour) defeats the "compact label"
#: intent: the full text still belongs somewhere, just not baked into the layout.
#: Truncated at a word boundary; the untruncated value is always in the hover tooltip.
LABEL_MAX_CHARS = 28

# Dark theme: readable on the near-black background both plain `graph.svg` (opened
# directly) and `graph.html` (this stage's wrapper, custom tooltips) render against.
BG_COLOR = "#181818"
NODE_FILL = "#2b2b2e"
NODE_BORDER = "#6b6b70"
NODE_TEXT = "#e8e8e8"
EDGE_COLOR = "#8a8a90"
EDGE_TEXT = "#c9c9cf"

#: Category color per spec 0004's attribute types, drawn one small box each on the
#: edge itself (Graphviz's HTML-like labels, not a post-render hack — see
#: `_edge_boxes_label`). Order here is the left-to-right order on the edge and the
#: order of the HTML legend `to_html` renders.
ATTR_COLORS = {
    "time": "#4a90d9",  # blue
    "place": "#b07cd9",  # violet
    "institution": "#57b894",  # green
    "action": "#e0c341",  # yellow
}
BOX_TEXT_COLOR = "#151515"
#: Kept shorter than LABEL_MAX_CHARS -- several boxes now sit side by side on one edge.
BOX_LABEL_MAX_CHARS = 20


def _matches_institution(relation: dict, substring: str) -> bool:
    needle = substring.lower()
    return any(
        a["attr_type"] == "institution" and needle in a["value"].lower()
        for a in relation["attributes"]
    )


def filter_relations(
    relations: list[dict],
    low_coverage_relations: list[dict],
    *,
    institution_contains: str,
    high_coverage_only: bool = False,
) -> tuple[list[dict], int]:
    """Matched relation records, plus a count skipped for a null participant qid.

    Spec 0006 §1/§2: both relation files are searched by default — a low-coverage
    split means thin *overall* token coverage, usually a missing ``action``, not
    that its ``institution`` attribute is any less real. ``high_coverage_only``
    restricts the search to ``relations.jsonl`` alone.
    """
    pool = relations if high_coverage_only else [*relations, *low_coverage_relations]
    matched = [r for r in pool if _matches_institution(r, institution_contains)]
    kept = [r for r in matched if all(p["qid"] for p in r["participants"])]
    skipped_null_participant = len(matched) - len(kept)
    return kept, skipped_null_participant


@dataclass
class GraphData:
    """Node labels and per-pair contributing relations, before DOT rendering."""

    nodes: dict[str, str] = field(default_factory=dict)  # qid -> label
    edges: dict[frozenset, list[dict]] = field(default_factory=dict)  # {qid, qid} -> relations


def build_graph(matched: list[dict], entities: dict[str, dict]) -> GraphData:
    """Spec 0006 §3/§4: one node per participant qid, edges collapsed per pair.

    A relation with participants outside a 2-person pair (shouldn't occur — spec
    0002's candidates are always pairs) is skipped defensively rather than crashing
    on an unexpected shape.
    """
    data = GraphData()
    for relation in matched:
        qids = [p["qid"] for p in relation["participants"]]
        for qid in qids:
            if qid not in data.nodes:
                entity = entities.get(qid)
                data.nodes[qid] = entity["canonical_name"] if entity else qid
        if len(qids) != 2 or qids[0] == qids[1]:
            continue
        pair = frozenset(qids)
        data.edges.setdefault(pair, []).append(relation)
    return data


def _edge_tooltip(contributing: list[dict]) -> str:
    """Sentence text only, one line per contributing relation — shown when hovering
    the edge line itself, as opposed to one of its attribute boxes.

    The attribute bag used to be appended here too (`[time=..., institution=...]`),
    but that's now exactly what the boxes already show colored on the graph —
    repeating it as text in the tooltip was redundant once the boxes existed.
    """
    return "\n".join(relation["sentence"] for relation in contributing)


def short_label(text: str, max_chars: int = LABEL_MAX_CHARS) -> str:
    """Truncate at a word boundary, never mid-word. The untruncated text is never
    lost — it's what `_edge_tooltip` puts in the hover-only detail."""
    if len(text) <= max_chars:
        return text
    head = text[:max_chars].rsplit(" ", 1)[0].rstrip(",.;:")
    return f"{head}…" if head else f"{text[:max_chars].rstrip()}…"


def _dot_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _html_escape(text: str) -> str:
    """Escaping for Graphviz's HTML-like labels — a different rule set than DOT's
    quoted-string escaping (`_dot_escape`): entity-escaped, not backslash-escaped."""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _primary_attribute_values(contributing: list[dict]) -> dict[str, str]:
    """attr_type -> value, restricted to the known box categories (`ATTR_COLORS`),
    read from the edge's *primary* (first) contributing relation only — collapsing
    several relations onto one edge (spec 0006 §4) means a pair can carry more
    attribute values than fit on the graph without clutter; every contributing
    relation's full sentence is still available via the edge-line tooltip
    (`_edge_tooltip`). First-seen value wins if a type repeats within one relation.
    """
    primary = contributing[0]
    by_type: dict[str, str] = {}
    for attr in primary["attributes"]:
        if attr["attr_type"] in ATTR_COLORS and attr["attr_type"] not in by_type:
            by_type[attr["attr_type"]] = attr["value"]
    return by_type


def edge_id(a: str, b: str) -> str:
    """Stable id shared by the DOT `id` attribute (-> the SVG element's `id`) and
    `run()`'s HTML tooltip lookup table, so the two stay in sync by construction."""
    return f"edge__{a}__{b}"


def node_id(qid: str) -> str:
    return f"node__{qid}"


def box_id(a: str, b: str, attr_type: str) -> str:
    """Id for one attribute box on one edge — distinct from `edge_id`, so hovering
    a specific box (its own SVG sub-element, via `ID=` on the DOT `<TD>`) can show
    that attribute's own tooltip instead of the whole edge's."""
    return f"box__{a}__{b}__{attr_type}"


#: Graphviz only wraps an HTML-like-label `<TD>` in an addressable SVG element when
#: it carries *both* `ID=` and an interactive attribute (`TOOLTIP=`/`HREF=`) — a
#: bare `ID=` alone is silently dropped, emitting no element at all. When it does
#: wrap one, it's an `<a>` prefixed `a_` onto the given id (verified empirically
#: against this machine's Graphviz 16.0.0; undocumented but stable behaviour). This
#: is *why* per-box hover used to fall back to the whole edge's tooltip no matter
#: which box was hovered — the boxes were never individually addressable elements
#: in the first place. `svg_box_id` is the id actually reachable in the DOM;
#: `box_id` (used for the DOT `ID=` value itself) is not.
_SVG_ANCHOR_PREFIX = "a_"


def svg_box_id(a: str, b: str, attr_type: str) -> str:
    return f"{_SVG_ANCHOR_PREFIX}{box_id(a, b, attr_type)}"


def box_tooltips_for_edge(a: str, b: str, contributing: list[dict]) -> dict[str, str]:
    """SVG id (`svg_box_id`) -> "Attr: full value" for every box `_edge_boxes_label`
    draws for this edge — derived the same way (`_primary_attribute_values`), so
    the ids always match what actually ends up in the DOT/SVG."""
    return {
        svg_box_id(a, b, attr_type): f"{attr_type.capitalize()}: {value}"
        for attr_type, value in _primary_attribute_values(contributing).items()
    }


def _edge_boxes_label(a: str, b: str, contributing: list[dict]) -> str:
    """One small color-coded box per attribute type, category-coded per
    `ATTR_COLORS` (action yellow, time blue, etc.). Each box gets its own `ID=`
    (-> `box_id`) *and* a `TOOLTIP=` — both are required for Graphviz to emit an
    addressable element at all (see `_SVG_ANCHOR_PREFIX`); the `TOOLTIP=` text also
    serves as a plain-`graph.svg` fallback (native, unstyled hover), the same
    graceful-degrade role the edge-level `tooltip` already plays.

    Returns Graphviz HTML-like label content (no outer `<...>` — `to_dot` adds
    that), or `""` if the primary relation has none of the known attribute types
    (shouldn't happen for a matched relation, which is guaranteed an `institution`
    hit by `filter_relations`, but handled rather than assumed).
    """
    by_type = _primary_attribute_values(contributing)
    if not by_type:
        return ""
    cells = "".join(
        f'<TD ID="{_html_escape(box_id(a, b, attr_type))}" '
        f'TOOLTIP="{_html_escape(attr_type.capitalize())}: '
        f'{_html_escape(by_type[attr_type])}" '
        f'BGCOLOR="{ATTR_COLORS[attr_type]}"><FONT COLOR="{BOX_TEXT_COLOR}" '
        f'POINT-SIZE="9">{_html_escape(short_label(by_type[attr_type], BOX_LABEL_MAX_CHARS))}'
        f"</FONT></TD>"
        for attr_type in ATTR_COLORS
        if attr_type in by_type
    )
    return (
        '<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="3" CELLPADDING="4">'
        f"<TR>{cells}</TR></TABLE>"
    )


def to_dot(data: GraphData) -> str:
    """Render `GraphData` as undirected DOT source. Node/edge order is sorted by
    qid so output is byte-stable across runs regardless of input order.

    ``overlap=false`` is Graphviz's own non-overlapping placement (`neato`/`fdp`
    otherwise let nodes sit on top of each other at this density) — the feature the
    HTML wrapper was specifically asked to use. Edge/node ``id`` is set explicitly
    so `run()`'s HTML post-processing can address the same elements Graphviz drew.
    The DOT ``tooltip`` attribute is kept as a plain-SVG fallback (native
    hover-on-title) for anyone who opens `graph.svg` directly rather than
    `graph.html`; the HTML wrapper strips it and substitutes a styled tooltip.
    """
    lines = [
        "graph relations {",
        f'  bgcolor="{BG_COLOR}";',
        "  overlap=false;",
        "  sep=\"+12\";",
        "  splines=true;",
        f'  node [shape=ellipse, style=filled, fillcolor="{NODE_FILL}", '
        f'color="{NODE_BORDER}", fontcolor="{NODE_TEXT}", fontname="sans-serif"];',
        f'  edge [color="{EDGE_COLOR}", fontcolor="{EDGE_TEXT}", fontname="sans-serif", '
        "fontsize=10];",
    ]
    for qid in sorted(data.nodes):
        lines.append(
            f'  "{_dot_escape(qid)}" [id="{_dot_escape(node_id(qid))}", '
            f'label="{_dot_escape(data.nodes[qid])}"];'
        )
    for pair in sorted(data.edges, key=lambda p: sorted(p)):
        a, b = sorted(pair)
        contributing = data.edges[pair]
        boxes = _edge_boxes_label(a, b, contributing)
        # HTML-like labels use `label=<...>` -- unquoted, angle-bracket delimited,
        # a different DOT construct from the quoted string `label="..."` used
        # everywhere else here. Falls back to an empty plain label in the
        # (shouldn't-happen, see `_edge_boxes_label`) case of no known attr_type.
        label_attr = f"label=<{boxes}>" if boxes else 'label=""'
        tooltip = _dot_escape(_edge_tooltip(contributing))
        lines.append(
            f'  "{_dot_escape(a)}" -- "{_dot_escape(b)}" '
            f'[id="{_dot_escape(edge_id(a, b))}", {label_attr}, tooltip="{tooltip}"];'
        )
    lines.append("}")
    return "\n".join(lines) + "\n"


_EDGE_TITLE_RE = re.compile(
    r'(<g id="edge__[^"]*" class="edge">)\s*<title>.*?</title>', re.DOTALL
)
#: Per-box native tooltips render differently from the edge's: an anchor-wrapped
#: cell (see `_SVG_ANCHOR_PREFIX`) carries its text as an `xlink:title="..."`
#: *attribute* on the `<a>`, not a nested `<title>` element — verified against this
#: machine's actual Graphviz output, not assumed from the edge case. Stripped the
#: same way: the plain-`graph.svg` fallback keeps it, `graph.html` replaces it.
_BOX_TITLE_RE = re.compile(r'(<a) xlink:title="[^"]*"')

_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
  html, body {{
    background: {bg}; color: {text}; margin: 0; padding: 0; height: 100%;
    overflow: hidden; font-family: sans-serif;
  }}
  #viz-canvas {{ position: relative; width: 100%; height: 100%; overflow: hidden; }}
  #viz-canvas svg {{
    position: absolute; top: 0; left: 0; transform-origin: 0 0; cursor: grab;
  }}
  #viz-canvas.dragging svg {{ cursor: grabbing; }}
  .viz-tooltip {{
    position: fixed; z-index: 1000; max-width: 420px;
    background: #101012; color: #f0f0f0; border: 1px solid #4a4a50;
    border-radius: 6px; padding: 8px 10px; font-size: 13px; line-height: 1.4;
    white-space: pre-line; pointer-events: none; box-shadow: 0 4px 16px rgba(0,0,0,0.5);
    opacity: 0; transition: opacity 0.1s ease-out;
  }}
  .viz-tooltip.visible {{ opacity: 1; }}
  g.edge {{ cursor: pointer; }}
  g.edge:hover path {{ stroke: {edge_hover}; stroke-width: 2; }}
  .viz-legend {{
    position: fixed; top: 12px; left: 12px; z-index: 1000;
    background: rgba(16,16,18,0.85); border: 1px solid #4a4a50; border-radius: 6px;
    padding: 8px 12px; font-size: 12px; display: flex; gap: 14px; align-items: center;
  }}
  .viz-legend .swatch {{
    display: inline-block; width: 10px; height: 10px; border-radius: 2px;
    margin-right: 5px; vertical-align: middle;
  }}
  .viz-hint {{
    position: fixed; bottom: 12px; left: 12px; z-index: 1000; font-size: 11px;
    color: #8a8a90;
  }}
</style>
</head>
<body>
<div id="viz-canvas">{svg}</div>
<div class="viz-legend">{legend}</div>
<div class="viz-hint">scroll to zoom &middot; drag to pan &middot; double-click to reset</div>
<div class="viz-tooltip" id="viz-tooltip"></div>
<script>
  // Delegated on the canvas, not per-element: a box sits *inside* its edge's <g>,
  // so the two would both fire "mouseenter" for the same pointer position with
  // independent per-element listeners -- delegation plus closest()'s inside-out
  // walk gives the box priority over its parent edge for free, and keeps working
  // as the pointer moves from a box onto the bare edge line without a stale
  // tooltip left over from whichever fired last.
  const TOOLTIPS = {tooltips_json};
  const tip = document.getElementById("viz-tooltip");
  const canvas = document.getElementById("viz-canvas");
  canvas.addEventListener("mousemove", function (e) {{
    const boxEl = e.target.closest('[id^="a_box__"]');
    const edgeEl = e.target.closest("g.edge[id]");
    const key = boxEl ? boxEl.id : (edgeEl ? edgeEl.id : null);
    const text = key ? TOOLTIPS[key] : null;
    if (text) {{
      tip.textContent = text;
      tip.classList.add("visible");
      tip.style.left = (e.clientX + 14) + "px";
      tip.style.top = (e.clientY + 14) + "px";
    }} else {{
      tip.classList.remove("visible");
    }}
  }});
  canvas.addEventListener("mouseleave", function () {{
    tip.classList.remove("visible");
  }});
</script>
<script>
  (function () {{
    const canvas = document.getElementById("viz-canvas");
    const svg = canvas.querySelector("svg");
    if (!svg) return;
    const MIN_SCALE = 0.1, MAX_SCALE = 8;
    const vb = svg.viewBox && svg.viewBox.baseVal;
    const svgW = (vb && vb.width) || svg.width.baseVal.value;
    const svgH = (vb && vb.height) || svg.height.baseVal.value;

    let scale = 1, tx = 0, ty = 0;
    function apply() {{
      svg.style.transform = "translate(" + tx + "px, " + ty + "px) scale(" + scale + ")";
    }}
    function fitToView() {{
      const cw = canvas.clientWidth, ch = canvas.clientHeight;
      scale = Math.max(MIN_SCALE, Math.min(MAX_SCALE,
        Math.min(cw / svgW, ch / svgH) * 0.92));
      tx = (cw - svgW * scale) / 2;
      ty = (ch - svgH * scale) / 2;
      apply();
    }}
    fitToView();

    canvas.addEventListener("wheel", function (e) {{
      e.preventDefault();
      const before = scale;
      const factor = e.deltaY < 0 ? 1.12 : 1 / 1.12;
      scale = Math.max(MIN_SCALE, Math.min(MAX_SCALE, scale * factor));
      tx = e.clientX - ((e.clientX - tx) / before) * scale;
      ty = e.clientY - ((e.clientY - ty) / before) * scale;
      apply();
    }}, {{ passive: false }});

    let dragging = false, lastX = 0, lastY = 0;
    canvas.addEventListener("mousedown", function (e) {{
      dragging = true; lastX = e.clientX; lastY = e.clientY;
      canvas.classList.add("dragging");
    }});
    window.addEventListener("mousemove", function (e) {{
      if (!dragging) return;
      tx += e.clientX - lastX; ty += e.clientY - lastY;
      lastX = e.clientX; lastY = e.clientY;
      apply();
    }});
    window.addEventListener("mouseup", function () {{
      dragging = false;
      canvas.classList.remove("dragging");
    }});
    canvas.addEventListener("dblclick", fitToView);
    window.addEventListener("resize", fitToView);
  }})();
</script>
</body>
</html>
"""


def _legend_html() -> str:
    return "".join(
        f'<span><i class="swatch" style="background:{color}"></i>{attr_type}</span>'
        for attr_type, color in ATTR_COLORS.items()
    )


def to_html(svg_text: str, tooltips: dict[str, str], *, title: str = "relations") -> str:
    """Wrap Graphviz's SVG in a dark-mode, zoomable/pannable HTML page with a
    styled hover tooltip and an attribute-color legend.

    Graphviz's own `tooltip` attribute renders as an SVG `<title>`, which the
    browser shows as a plain, unstyled, delayed OS tooltip that can't wrap long
    text sensibly — exactly what was asked to replace. Each matching edge's
    `<title>` is stripped (native fallback stays intact in the plain `graph.svg`
    this HTML is built from) and reattached as a `mouseenter`/`mousemove`-driven
    `<div>`, keyed by the same `id` `to_dot` gave the edge.

    Zoom/pan is hand-rolled (mouse wheel scales around the cursor, drag pans, a
    double-click refits the whole graph) rather than pulling in a JS library —
    the page stays a single self-contained file with no external/CDN dependency,
    consistent with this project's offline-reproducibility posture elsewhere
    (e.g. the git-tracked Wikidata cache, spec 0002).
    """
    svg_no_titles = _EDGE_TITLE_RE.sub(r"\1", svg_text)
    svg_no_titles = _BOX_TITLE_RE.sub(r"\1", svg_no_titles)
    # A tooltip's sentence text is untrusted-ish (extracted prose) and lands inside an
    # inline <script>; the HTML tokenizer ends the block on a literal "</" regardless
    # of JS string quoting, so it must be escaped before embedding, not just JSON-quoted.
    tooltips_json = json.dumps(tooltips, ensure_ascii=False).replace("</", "<\\/")
    return _HTML_TEMPLATE.format(
        title=title,
        bg=BG_COLOR,
        text=NODE_TEXT,
        edge_hover="#ffffff",
        svg=svg_no_titles,
        legend=_legend_html(),
        tooltips_json=tooltips_json,
    )


def _slug(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug or "graph"


def resolve_engine(engine: str, graphviz_bin: str | None) -> str:
    """Find the engine executable, per spec 0006 §5 (config over a hardcoded PATH
    assumption — this machine's Graphviz install isn't on PATH)."""
    if graphviz_bin:
        for name in (f"{engine}.exe", engine):
            candidate = Path(graphviz_bin) / name
            if candidate.exists():
                return str(candidate)
        raise FileNotFoundError(f"{engine!r} not found in {graphviz_bin}")
    found = shutil.which(engine)
    if not found:
        raise FileNotFoundError(
            f"{engine!r} not found on PATH — pass --graphviz-bin or set "
            "INPNET_GRAPHVIZ_BIN to the Graphviz bin directory"
        )
    return found


def run(
    relations_dir: Path,
    entities_path: Path,
    out_dir: Path,
    *,
    institution_contains: str,
    name: str | None = None,
    high_coverage_only: bool = False,
    engine: str = DEFAULT_ENGINE,
    graphviz_bin: str | None = None,
) -> dict[str, Any]:
    """Filter -> build graph -> write `.dot` -> render `.svg` -> write manifest."""
    relations_path = relations_dir / "relations.jsonl"
    low_coverage_path = relations_dir / "low_coverage_relations.jsonl"

    relations = read_jsonl(relations_path)
    low_coverage = [] if high_coverage_only else read_jsonl(low_coverage_path)
    entities = {e["qid"]: e for e in read_jsonl(entities_path)}

    matched, skipped_null_participant = filter_relations(
        relations,
        low_coverage,
        institution_contains=institution_contains,
        high_coverage_only=high_coverage_only,
    )
    data = build_graph(matched, entities)
    dot_text = to_dot(data)

    run_dir = out_dir / (name or _slug(institution_contains))
    run_dir.mkdir(parents=True, exist_ok=True)
    dot_path = run_dir / "graph.dot"
    svg_path = run_dir / "graph.svg"
    html_path = run_dir / "graph.html"
    dot_path.write_text(dot_text, encoding="utf-8", newline="\n")

    engine_exe = resolve_engine(engine, graphviz_bin)
    subprocess.run(
        [engine_exe, "-Tsvg", str(dot_path), "-o", str(svg_path)], check=True
    )

    tooltips: dict[str, str] = {}
    for pair, contributing in data.edges.items():
        a, b = sorted(pair)
        tooltips[edge_id(a, b)] = _edge_tooltip(contributing)
        tooltips.update(box_tooltips_for_edge(a, b, contributing))
    svg_text = svg_path.read_text(encoding="utf-8")
    html_path.write_text(
        to_html(svg_text, tooltips, title=f"relations: {institution_contains}"),
        encoding="utf-8",
        newline="\n",
    )

    counts = {
        "matched_relations": len(matched),
        "skipped_null_participant": skipped_null_participant,
        "nodes": len(data.nodes),
        "edges": len(data.edges),
    }
    write_manifest(
        run_dir,
        stage="render-graph",
        spec="0006-graph-visualization",
        config={
            "institution_contains": institution_contains,
            "high_coverage_only": high_coverage_only,
            "engine": engine,
        },
        counts=counts,
        inputs={
            "relations": relations_path,
            "low_coverage_relations": low_coverage_path,
            "entities": entities_path,
        },
    )
    return counts
