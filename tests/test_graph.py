"""Tests for the institution-filtered graph visualization. Implements spec 0006."""

from __future__ import annotations

import shutil

import pytest

from inpnet.viz.graph import (
    ATTR_COLORS,
    BG_COLOR,
    LABEL_MAX_CHARS,
    NODE_FILL,
    NODE_TEXT,
    GraphData,
    _edge_tooltip,
    box_id,
    box_tooltips_for_edge,
    build_graph,
    edge_id,
    filter_relations,
    node_id,
    resolve_engine,
    short_label,
    svg_box_id,
    to_dot,
    to_html,
)


def _relation(
    relation_id, head, tail, *, sentence="They worked together.", attributes=None
):
    return {
        "relation_id": relation_id,
        "sentence_id": "Q1:0:0",
        "sentence": sentence,
        "participants": [{"mention_id": "m1", "qid": head}, {"mention_id": "m2", "qid": tail}],
        "attributes": attributes or [],
        "coverage": 1.0,
        "pronoun_resolved": False,
    }


def _institution(value):
    return {"attr_type": "institution", "value": value, "span": [0, 4], "source": "spacy_ner"}


def _action(value):
    return {"attr_type": "action", "value": value, "span": [0, 4], "source": "dep_parse"}


def _time(value):
    return {"attr_type": "time", "value": value, "span": [0, 4], "source": "spacy_ner"}


def _place(value):
    return {"attr_type": "place", "value": value, "span": [0, 4], "source": "spacy_ner"}


ENTITIES = {
    "Q1": {"qid": "Q1", "canonical_name": "Alice Example"},
    "Q2": {"qid": "Q2", "canonical_name": "Bob Sample"},
    "Q3": {"qid": "Q3", "canonical_name": "Carol Instance"},
}


class TestFilterRelations:
    def test_substring_match_is_case_insensitive(self):
        relations = [
            _relation("r1", "Q1", "Q2", attributes=[_institution("Yale University")]),
            _relation("r2", "Q1", "Q3", attributes=[_institution("YALE MEDICAL SCHOOL")]),
            _relation("r3", "Q2", "Q3", attributes=[_institution("Harvard University")]),
        ]
        matched, _ = filter_relations([], relations, institution_contains="yale")
        assert {r["relation_id"] for r in matched} == {"r1", "r2"}

    def test_high_coverage_only_excludes_low_coverage_file(self):
        high = [_relation("r1", "Q1", "Q2", attributes=[_institution("Yale")])]
        low = [_relation("r2", "Q1", "Q3", attributes=[_institution("Yale")])]

        both, _ = filter_relations(high, low, institution_contains="yale")
        assert {r["relation_id"] for r in both} == {"r1", "r2"}

        only_high, _ = filter_relations(
            high, low, institution_contains="yale", high_coverage_only=True
        )
        assert {r["relation_id"] for r in only_high} == {"r1"}

    def test_relation_with_null_participant_is_skipped_and_counted(self):
        relations = [_relation("r1", "Q1", None, attributes=[_institution("Yale")])]
        matched, skipped = filter_relations([], relations, institution_contains="yale")
        assert matched == []
        assert skipped == 1

    def test_non_matching_attr_type_is_ignored(self):
        """A `place` value containing "yale" must not match the institution filter."""
        relations = [_relation("r1", "Q1", "Q2", attributes=[
            {"attr_type": "place", "value": "Yale, Oklahoma", "span": [0, 4], "source": "spacy_ner"}
        ])]
        matched, _ = filter_relations([], relations, institution_contains="yale")
        assert matched == []


class TestBuildGraph:
    def test_nodes_deduplicated_and_labeled_from_entities(self):
        relations = [
            _relation("r1", "Q1", "Q2"),
            _relation("r2", "Q1", "Q3"),
        ]
        data = build_graph(relations, ENTITIES)
        assert data.nodes == {"Q1": "Alice Example", "Q2": "Bob Sample", "Q3": "Carol Instance"}

    def test_unknown_qid_falls_back_to_qid_as_label(self):
        data = build_graph([_relation("r1", "Q1", "Q99")], ENTITIES)
        assert data.nodes["Q99"] == "Q99"

    def test_multiple_relations_between_same_pair_collapse_to_one_edge(self):
        relations = [
            _relation("r1", "Q1", "Q2", sentence="First sentence.", attributes=[_action("met")]),
            _relation("r2", "Q2", "Q1", sentence="Second sentence.", attributes=[_action("collaborated")]),
        ]
        data = build_graph(relations, ENTITIES)
        assert len(data.edges) == 1
        (contributing,) = data.edges.values()
        assert len(contributing) == 2
        assert {r["relation_id"] for r in contributing} == {"r1", "r2"}

    def test_distinct_pairs_produce_distinct_edges(self):
        relations = [_relation("r1", "Q1", "Q2"), _relation("r2", "Q1", "Q3")]
        data = build_graph(relations, ENTITIES)
        assert len(data.edges) == 2


class TestToDot:
    def test_output_is_undirected_graph_with_nodes_and_edges(self):
        data = build_graph([_relation("r1", "Q1", "Q2")], ENTITIES)
        text = to_dot(data)
        assert text.startswith("graph relations {")
        assert 'label="Alice Example"' in text
        assert '"Q1" [id="node__Q1"' in text
        assert '"Q1" -- "Q2"' in text

    def test_layout_is_configured_for_non_overlapping_placement(self):
        """Regression: neato/fdp allow nodes to sit on top of each other unless
        told otherwise -- `overlap=false` is what was specifically asked for."""
        text = to_dot(build_graph([_relation("r1", "Q1", "Q2")], ENTITIES))
        assert "overlap=false" in text

    def test_dark_theme_colors_are_set(self):
        text = to_dot(build_graph([_relation("r1", "Q1", "Q2")], ENTITIES))
        assert f'bgcolor="{BG_COLOR}"' in text
        assert NODE_FILL in text
        assert NODE_TEXT in text

    def test_quote_in_label_is_escaped(self):
        entities = {**ENTITIES, "Q1": {"qid": "Q1", "canonical_name": 'Alice "Al" Example'}}
        data = build_graph([_relation("r1", "Q1", "Q2")], entities)
        text = to_dot(data)
        assert 'label="Alice \\"Al\\" Example"' in text
        # and the file must still be syntactically balanced -- no bare unescaped quote
        assert text.count('\\"') == 2

    def test_tooltip_carries_every_contributing_relation(self):
        relations = [
            _relation("r1", "Q1", "Q2", sentence="First sentence.", attributes=[_action("met")]),
            _relation("r2", "Q2", "Q1", sentence="Second sentence.", attributes=[_action("collaborated")]),
        ]
        data = build_graph(relations, ENTITIES)
        text = to_dot(data)
        assert "First sentence." in text
        assert "Second sentence." in text

    def test_edge_line_tooltip_no_longer_repeats_the_attribute_bag(self):
        """Regression: the tooltip used to append "[time=..., institution=...]"
        after the sentence -- exactly what the colored boxes now already show on
        the graph, making it redundant. `_edge_tooltip` is sentence text only."""
        relations = [
            _relation(
                "r1", "Q1", "Q2", sentence="He joined in 1995.",
                attributes=[_time("1995"), _institution("Yale")],
            )
        ]
        tooltip = _edge_tooltip(relations)
        assert tooltip == "He joined in 1995."
        assert "time=" not in tooltip
        assert "institution=" not in tooltip
        assert "[" not in tooltip

    def test_output_is_deterministic_regardless_of_input_order(self):
        relations_a = [_relation("r1", "Q2", "Q1"), _relation("r2", "Q3", "Q1")]
        relations_b = [_relation("r2", "Q3", "Q1"), _relation("r1", "Q2", "Q1")]
        text_a = to_dot(build_graph(relations_a, ENTITIES))
        text_b = to_dot(build_graph(relations_b, ENTITIES))
        assert text_a == text_b

    def test_empty_graph_still_closes_the_brace(self):
        text = to_dot(GraphData())
        assert text.startswith("graph relations {")
        assert text.endswith("}\n")
        assert "--" not in text  # no edge lines when there are no edges

    def test_long_action_is_shortened_on_the_graph_but_full_text_reaches_its_box_tooltip(self):
        """Regression: spec 0004 measured `action` spans averaging 91 characters —
        drawing one directly on the graph as the label defeats "compact label". The
        *visible* (`FONT`) text is truncated; the untruncated value still lives in
        the DOT too, as that box's own `TOOLTIP=` (the plain-`graph.svg` fallback,
        see `_edge_boxes_label`) -- not just via `box_tooltips_for_edge`, which
        `run()` uses for the styled `graph.html` version of the same content."""
        long_action = (
            "was appointed to a distinguished chair in theoretical physics following "
            "a decade of postdoctoral research"
        )
        relations = [_relation("r1", "Q1", "Q2", attributes=[_action(long_action)])]
        text = to_dot(build_graph(relations, ENTITIES))
        assert long_action in text  # present, in that box's own TOOLTIP=
        visible_text = text.split("<FONT")[1].split("</FONT>")[0]
        assert long_action not in visible_text  # but not what's actually drawn

        tooltips = box_tooltips_for_edge("Q1", "Q2", relations)
        assert tooltips[svg_box_id("Q1", "Q2", "action")] == f"Action: {long_action}"

    def test_edge_label_is_an_html_like_table_of_colored_boxes(self):
        relations = [
            _relation(
                "r1", "Q1", "Q2",
                attributes=[_institution("Yale University"), _action("supervised"), _time("1995")],
            )
        ]
        text = to_dot(build_graph(relations, ENTITIES))
        assert "label=<" in text  # HTML-like label, not a quoted string
        assert f'BGCOLOR="{ATTR_COLORS["institution"]}"' in text
        assert f'BGCOLOR="{ATTR_COLORS["action"]}"' in text
        assert f'BGCOLOR="{ATTR_COLORS["time"]}"' in text
        assert f'BGCOLOR="{ATTR_COLORS["place"]}"' not in text  # none in this relation

    def test_box_has_both_id_and_tooltip_dot_attributes(self):
        """Regression: Graphviz silently drops a bare `ID=` on an HTML-like-label
        `<TD>` -- no SVG element is emitted for it at all -- unless the cell also
        carries `TOOLTIP=` (or `HREF=`). Without this, every box was invisible to
        the hover JS and the whole edge's (or a neighboring box's) tooltip showed
        instead, regardless of which box was actually under the cursor. Both
        attributes must be present together."""
        relations = [_relation("r1", "Q1", "Q2", attributes=[_action("supervised")])]
        text = to_dot(build_graph(relations, ENTITIES))
        assert f'ID="{box_id("Q1", "Q2", "action")}"' in text
        assert 'TOOLTIP="Action: supervised"' in text

    def test_svg_box_id_is_the_dot_box_id_with_graphviz_anchor_prefix(self):
        assert svg_box_id("Q1", "Q2", "action") == "a_" + box_id("Q1", "Q2", "action")

    def test_box_values_are_html_escaped(self):
        relations = [_relation("r1", "Q1", "Q2", attributes=[_institution('R&D <Lab>')])]
        text = to_dot(build_graph(relations, ENTITIES))
        assert "R&amp;D &lt;Lab&gt;" in text
        assert "R&D <Lab>" not in text.split("tooltip=")[0]

    def test_boxes_come_from_the_primary_relation_only(self):
        """Two relations collapse onto one edge (spec 0006 §4); the drawn boxes,
        and their tooltips, reflect only the first one -- the second relation's
        own attribute value is still reachable, just via its full sentence in the
        edge-line tooltip (`_edge_tooltip`), not duplicated as a second box."""
        relations = [
            _relation(
                "r1", "Q1", "Q2", sentence="Joined Yale in 1990.",
                attributes=[_institution("Yale"), _time("1990")],
            ),
            _relation(
                "r2", "Q2", "Q1", sentence="Later promoted to professor in 2005.",
                attributes=[_institution("Yale"), _time("2005")],
            ),
        ]
        text = to_dot(build_graph(relations, ENTITIES))
        label_part = text.split("tooltip=")[0]
        assert "1990" in label_part
        assert "2005" not in label_part  # not drawn as a second box

        edge_tooltip = _edge_tooltip(relations)
        assert "2005" in edge_tooltip  # recoverable via the second sentence's own text

        box_tooltips = box_tooltips_for_edge("Q1", "Q2", relations)
        assert box_tooltips == {svg_box_id("Q1", "Q2", "institution"): "Institution: Yale",
                                 svg_box_id("Q1", "Q2", "time"): "Time: 1990"}


class TestShortLabel:
    def test_short_text_is_unchanged(self):
        assert short_label("worked with") == "worked with"

    def test_long_text_is_truncated_at_a_word_boundary(self):
        text = "was appointed to a distinguished chair in theoretical physics"
        result = short_label(text, max_chars=28)
        assert len(result) <= 29  # 28 + the ellipsis character
        assert not result.endswith(" …")
        assert text.startswith(result.rstrip("…").rstrip())

    def test_never_exceeds_max_chars_even_with_no_spaces(self):
        result = short_label("a" * 50, max_chars=10)
        assert len(result) <= 11


class TestToHtml:
    def test_page_is_dark_themed(self):
        html = to_html("<svg></svg>", {})
        assert BG_COLOR in html

    def test_native_edge_title_is_stripped(self):
        svg = '<g id="edge__Q1__Q2" class="edge"><title>Q1--Q2</title><path/></g>'
        html = to_html(svg, {"edge__Q1__Q2": "full sentence detail"})
        assert "<title>Q1--Q2</title>" not in html
        assert 'id="edge__Q1__Q2" class="edge">' in html

    def test_tooltip_text_is_embedded_for_javascript(self):
        html = to_html("<svg></svg>", {"edge__Q1__Q2": "the full sentence, with a comma"})
        assert "the full sentence, with a comma" in html

    def test_tooltip_json_is_safely_escaped(self):
        """A sentence containing `</script>` must not break out of the <script>
        tag — the HTML tokenizer ends it on the literal text regardless of JS
        string quoting, so JSON-quoting alone (which doesn't escape "/") isn't
        enough."""
        html = to_html("<svg></svg>", {"edge__Q1__Q2": 'sentence with </script> inside'})
        assert "</script> inside" not in html
        assert "<\\/script> inside" in html

    def test_page_has_wheel_zoom_and_drag_pan(self):
        html = to_html("<svg></svg>", {})
        assert 'addEventListener("wheel"' in html
        assert 'addEventListener("mousedown"' in html
        assert 'addEventListener("dblclick"' in html

    def test_zoom_pan_has_no_external_script_dependency(self):
        """Self-contained on purpose -- see `to_html`'s docstring."""
        html = to_html("<svg></svg>", {})
        assert "<script src=" not in html
        assert "cdn." not in html.lower()

    def test_legend_lists_every_attribute_category_with_its_color(self):
        html = to_html("<svg></svg>", {})
        for attr_type, color in ATTR_COLORS.items():
            assert attr_type in html
            assert color in html

    def test_tooltip_lookup_checks_a_box_before_falling_back_to_its_edge(self):
        """Regression: with one listener per element, hovering a box (nested
        inside its edge's <g>) used to leave the *edge's* tooltip showing instead
        of the more specific box's own -- delegation with box-first `closest()`
        fixes that. Assert the box selector is checked, and checked before the
        edge one, rather than re-deriving actual hover behaviour (covered live in
        the browser, see spec 0006's Acceptance criteria)."""
        html = to_html("<svg></svg>", {})
        script = html.split('canvas.addEventListener("mousemove"', 1)[1]
        box_check = script.index('closest(\'[id^="a_box__"]\')')
        edge_check = script.index('closest("g.edge[id]")')
        assert box_check < edge_check

    def test_single_delegated_listener_not_one_per_edge(self):
        """The old per-element-listener version queried every `g.edge[id]` and
        attached three listeners each; delegation replaces that with one
        listener on the canvas."""
        html = to_html("<svg></svg>", {})
        assert "querySelectorAll" not in html


class TestEdgeAndNodeIds:
    def test_ids_are_stable_and_distinct(self):
        assert edge_id("Q1", "Q2") != edge_id("Q1", "Q3")
        assert node_id("Q1") != node_id("Q2")
        assert edge_id("Q1", "Q2") == edge_id("Q1", "Q2")


class TestResolveEngine:
    def test_missing_engine_in_given_dir_raises_clear_error(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="not found in"):
            resolve_engine("neato", str(tmp_path))

    def test_missing_engine_on_path_mentions_the_escape_hatch(self, monkeypatch):
        monkeypatch.setattr(shutil, "which", lambda _: None)
        with pytest.raises(FileNotFoundError, match="graphviz-bin"):
            resolve_engine("neato", None)

    def test_finds_engine_in_given_directory(self, tmp_path):
        exe = tmp_path / "neato.exe"
        exe.write_text("", encoding="utf-8")
        assert resolve_engine("neato", str(tmp_path)) == str(exe)

    @pytest.mark.skipif(shutil.which("neato") is None, reason="graphviz not on PATH")
    def test_finds_engine_on_path(self):
        assert resolve_engine("neato", None)
