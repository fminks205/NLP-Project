"""Tests for the seed stage. Implements spec 0001 §Decision 1."""

from inpnet.corpus.seed import (
    collapse_rows,
    qid_from_entity_url,
    title_from_article_url,
)


def test_title_from_article_url_decodes_and_unslugs():
    assert title_from_article_url("https://en.wikipedia.org/wiki/Niels_Bohr") == "Niels Bohr"


def test_title_from_article_url_handles_percent_encoding():
    # Real seed entries carry non-ASCII names; Schrödinger is the canonical case.
    url = "https://en.wikipedia.org/wiki/Erwin_Schr%C3%B6dinger"
    assert title_from_article_url(url) == "Erwin Schrödinger"


def test_qid_from_entity_url():
    assert qid_from_entity_url("http://www.wikidata.org/entity/Q71296") == "Q71296"


def _row(qid, article, **extra):
    row = {
        "person": f"http://www.wikidata.org/entity/{qid}",
        "article": f"https://en.wikipedia.org/wiki/{article}",
    }
    row.update(extra)
    return row


def test_collapse_rows_deduplicates_people():
    """OPTIONAL clauses multiply rows: 18,178 rows for 15,158 people (2026-08-12)."""
    rows = [
        _row("Q1", "A", birthYear="1900"),
        _row("Q1", "A", birthYear="1901"),  # Wikidata holds two birth dates
        _row("Q2", "B"),
    ]
    people = collapse_rows(rows)
    assert [p["qid"] for p in people] == ["Q1", "Q2"]
    assert people[0]["_variants"] == 1


def test_collapse_rows_keeps_first_value_and_fills_gaps():
    rows = [
        _row("Q1", "A", birthYear="1900"),
        _row("Q1", "A", birthYear="1901", deathYear="1980"),
    ]
    (person,) = collapse_rows(rows)
    assert person["birth_year"] == 1900, "first value wins, later rows must not overwrite"
    assert person["death_year"] == 1980, "a gap a later row covers should be filled"


def test_collapse_rows_is_deterministic():
    """Output must be stable across runs — the spec requires reproducibility."""
    rows = [_row("Q9", "Z"), _row("Q1", "A"), _row("Q5", "M")]
    assert [p["qid"] for p in collapse_rows(rows)] == ["Q1", "Q5", "Q9"]


def test_collapse_rows_tolerates_missing_optionals():
    (person,) = collapse_rows([_row("Q1", "A")])
    assert person["birth_year"] is None
    assert person["gender"] is None
