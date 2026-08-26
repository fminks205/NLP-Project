"""Tests for Wikidata resolution. Implements spec 0002 §Decision 2."""

import json

from inpnet.nlp.resolve import collect_link_targets
from inpnet.wiki import _claim_ids, _claim_year


def _doc(doc_id, links):
    return {
        "doc_id": doc_id,
        "qid": doc_id,
        "title": doc_id,
        "sections": [{"heading": "", "text": "x", "links": links}],
    }


def _link(target):
    return {"start": 0, "end": 1, "surface": "x", "target_title": target}


def test_collect_link_targets_counts_instances(tmp_path):
    path = tmp_path / "documents.jsonl"
    path.write_text(
        "\n".join(
            json.dumps(d)
            for d in [
                _doc("Q1", [_link("Niels Bohr"), _link("Physics")]),
                _doc("Q2", [_link("Niels Bohr")]),
            ]
        ),
        encoding="utf-8",
    )
    counts = collect_link_targets(path)
    assert counts["Niels Bohr"] == 2
    assert counts["Physics"] == 1


def test_collect_link_targets_handles_empty_lines(tmp_path):
    path = tmp_path / "documents.jsonl"
    path.write_text(json.dumps(_doc("Q1", [_link("A")])) + "\n\n", encoding="utf-8")
    assert collect_link_targets(path)["A"] == 1


# -- claim parsing -------------------------------------------------------

def _claim(value):
    return {"mainsnak": {"datavalue": {"value": value}}}


def test_claim_ids_extracts_entity_ids():
    claims = {"P31": [_claim({"id": "Q5"}), _claim({"id": "Q215627"})]}
    assert _claim_ids(claims, "P31") == {"Q5", "Q215627"}


def test_claim_ids_ignores_novalue_snaks():
    """Wikidata uses snaks with no datavalue for 'unknown' / 'no value'."""
    claims = {"P31": [{"mainsnak": {"snaktype": "novalue"}}, _claim({"id": "Q5"})]}
    assert _claim_ids(claims, "P31") == {"Q5"}


def test_claim_ids_missing_property_is_empty():
    assert _claim_ids({}, "P31") == set()


def test_claim_ids_ignores_deprecated_rank():
    """Regression: Q17021508 is a helicopter with a deprecated P31 = Q5.

    Reading every claim regardless of rank classified it as a person. Found by
    cross-checking against SPARQL, whose `wdt:` prefix applies this rule natively.
    """
    claims = {
        "P31": [
            {"rank": "deprecated", "mainsnak": {"datavalue": {"value": {"id": "Q5"}}}},
            {"rank": "normal", "mainsnak": {"datavalue": {"value": {"id": "Q16145172"}}}},
        ]
    }
    assert _claim_ids(claims, "P31") == {"Q16145172"}


def test_claim_ids_prefers_preferred_rank():
    claims = {
        "P31": [
            {"rank": "normal", "mainsnak": {"datavalue": {"value": {"id": "Q1"}}}},
            {"rank": "preferred", "mainsnak": {"datavalue": {"value": {"id": "Q2"}}}},
        ]
    }
    assert _claim_ids(claims, "P31") == {"Q2"}


def test_claim_year_prefers_preferred_rank():
    """Diophantus (Q178217) has normal +0201 and preferred +0200."""
    claims = {
        "P569": [
            {"rank": "normal", "mainsnak": {"datavalue": {"value": {"time": "+0201-00-00T00:00:00Z"}}}},
            {"rank": "preferred", "mainsnak": {"datavalue": {"value": {"time": "+0200-01-01T00:00:00Z"}}}},
        ]
    }
    assert _claim_year(claims, "P569") == 200


def test_claim_year_parses_wikidata_time():
    claims = {"P569": [_claim({"time": "+1885-10-07T00:00:00Z"})]}
    assert _claim_year(claims, "P569") == 1885


def test_claim_year_handles_bce_dates():
    """BCE years carry a leading minus; the sign must survive."""
    claims = {"P569": [_claim({"time": "-0384-01-01T00:00:00Z"})]}
    assert _claim_year(claims, "P569") == -384


def test_claim_year_returns_none_when_absent():
    assert _claim_year({}, "P569") is None


def test_claim_year_ignores_non_time_values():
    claims = {"P569": [_claim({"id": "Q5"})]}
    assert _claim_year(claims, "P569") is None
