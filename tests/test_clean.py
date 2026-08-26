"""Tests for the clean stage. Implements spec 0001 §Decision 3.

Fixtures use real Parsoid markup shapes taken from fetched physicist articles.
"""

from inpnet.corpus.clean import parse_html, verify_offsets


def _wrap(body: str) -> str:
    return f'<html><body><section data-mw-section-id="0">{body}</section></body></html>'


def test_extracts_prose_and_wikilink():
    html = _wrap(
        '<p>David Charles Lowe <a rel="mw:WikiLink" href="./Companion_of_the_Royal_Society">'
        "CRSNZ</a> is a New Zealand scientist.</p>"
    )
    (section,) = parse_html(html)
    assert section["text"] == "David Charles Lowe CRSNZ is a New Zealand scientist."
    (link,) = section["links"]
    assert link["surface"] == "CRSNZ"
    assert link["target_title"] == "Companion of the Royal Society"
    assert section["text"][link["start"] : link["end"]] == "CRSNZ"


def test_link_span_excludes_preceding_whitespace():
    """Regression: spans used to swallow the space before the anchor.

    The offsets stayed self-consistent, so a round-trip check passed while both
    the offsets and the surface were wrong by one character.
    """
    html = _wrap('<p>worked at <a rel="mw:WikiLink" href="./Bell_Labs">Bell Labs</a> later.</p>')
    (section,) = parse_html(html)
    (link,) = section["links"]
    assert link["surface"] == "Bell Labs"
    assert not link["surface"].startswith(" ")
    assert section["text"][link["start"]] == "B"


def test_link_span_excludes_whitespace_from_inside_the_anchor():
    """Regression: whitespace can arrive from within the anchor element."""
    html = _wrap(
        '<p>joined the <a rel="mw:WikiLink" href="./Kaiser_Wilhelm_Institute">'
        "<span> Kaiser Wilhelm Institute </span></a> in 1925.</p>"
    )
    (section,) = parse_html(html)
    (link,) = section["links"]
    assert link["surface"] == "Kaiser Wilhelm Institute"
    assert verify_offsets([section]) == []


def test_verify_offsets_flags_whitespace_spans():
    """The checker must catch shifted-but-consistent spans, not just mismatches."""
    section = {
        "heading": "",
        "text": "worked at Bell Labs later.",
        "links": [{"start": 9, "end": 19, "surface": " Bell Labs", "target_title": "Bell Labs"}],
    }
    problems = verify_offsets([section])
    assert len(problems) == 1
    assert "whitespace" in problems[0]


def test_verify_offsets_flags_mismatch():
    section = {
        "heading": "",
        "text": "worked at Bell Labs",
        "links": [{"start": 0, "end": 6, "surface": "Bell Labs", "target_title": "Bell Labs"}],
    }
    assert "offset mismatch" in verify_offsets([section])[0]


def test_offsets_survive_multiple_paragraphs():
    html = _wrap(
        '<p>First <a rel="mw:WikiLink" href="./One">One</a> here.</p>'
        '<p>Second <a rel="mw:WikiLink" href="./Two">Two</a> there.</p>'
    )
    (section,) = parse_html(html)
    assert "\n\n" in section["text"]
    assert verify_offsets([section]) == []
    assert [l["surface"] for l in section["links"]] == ["One", "Two"]


def test_reference_markers_are_removed():
    html = _wrap('<p>A claim<sup class="mw-ref"><a href="#cite_note-1">[1]</a></sup> follows.</p>')
    (section,) = parse_html(html)
    assert section["text"] == "A claim follows."
    assert section["links"] == []


def test_infobox_and_table_are_dropped():
    html = _wrap(
        '<table class="infobox"><tr><td>Born</td><td>1885</td></tr></table>'
        "<p>Real prose.</p>"
    )
    (section,) = parse_html(html)
    assert section["text"] == "Real prose."


def test_reference_sections_are_dropped():
    html = (
        '<html><body><section data-mw-section-id="0"><p>Lead prose.</p></section>'
        '<section data-mw-section-id="1"><h2>Career</h2><p>Career prose.</p></section>'
        '<section data-mw-section-id="2"><h2>References</h2><p>Citation junk.</p></section>'
        '<section data-mw-section-id="3"><h2>See also</h2><p>Other links.</p></section>'
        "</body></html>"
    )
    sections = parse_html(html)
    assert [s["heading"] for s in sections] == ["", "Career"]


def test_external_links_are_not_recorded_as_entity_links():
    html = _wrap('<p>See <a rel="mw:ExtLink" href="https://example.org">the site</a>.</p>')
    (section,) = parse_html(html)
    assert section["links"] == []


def test_fragment_only_links_are_ignored():
    html = _wrap('<p>See <a rel="mw:WikiLink" href="#section">below</a>.</p>')
    (section,) = parse_html(html)
    assert section["links"] == []


def test_link_with_fragment_keeps_base_title():
    html = _wrap('<p>See <a rel="mw:WikiLink" href="./Physics#History">physics</a>.</p>')
    (section,) = parse_html(html)
    assert section["links"][0]["target_title"] == "Physics"


def test_redlink_query_string_is_stripped():
    """Regression: red links carry ?action=edit&redlink=1 in the href.

    Left in place the target became 'Wolfgang Riezler?action=edit&redlink=1', which
    matches nothing in Wikidata. 4.5% of the corpus's link targets were affected.
    """
    html = _wrap(
        '<p>with <a rel="mw:WikiLink" href="./Wolfgang_Riezler?action=edit&amp;redlink=1">'
        "Wolfgang Riezler</a> in Munich.</p>"
    )
    (section,) = parse_html(html)
    assert section["links"][0]["target_title"] == "Wolfgang Riezler"


def test_underscores_and_encoding_in_targets():
    html = _wrap('<p>Met <a rel="mw:WikiLink" href="./Erwin_Schr%C3%B6dinger">him</a>.</p>')
    (section,) = parse_html(html)
    assert section["links"][0]["target_title"] == "Erwin Schrödinger"


def test_empty_sections_are_skipped():
    html = _wrap('<table class="infobox"><tr><td>only a table</td></tr></table>')
    assert parse_html(html) == []
