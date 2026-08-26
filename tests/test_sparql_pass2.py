"""Tests for the SPARQL pass 2 of resolve. Implements spec 0002 §Decision 2."""

import pytest

from inpnet.nlp.resolve import (
    CHUNK_FLOOR,
    _chunked_query,
    humans_via_sparql,
    metadata_via_sparql,
    values_clause,
)
from inpnet.wiki import QueryTruncated, WikiClient


def test_values_clause_renders_qids():
    assert values_clause(["Q1", "Q2"]) == "wd:Q1 wd:Q2"


def test_values_clause_empty():
    assert values_clause([]) == ""


class _FakeClient:
    """Records the queries it was asked to run and replays scripted results."""

    def __init__(self, responder):
        self.responder = responder
        self.queries: list[str] = []

    def sparql(self, query, **kwargs):
        self.queries.append(query)
        return self.responder(query)


def test_chunked_query_covers_every_id_exactly_once():
    seen = []

    def responder(query):
        seen.extend(q for q in query.split() if q.startswith("wd:"))
        return []

    client = _FakeClient(responder)
    ids = [f"Q{i}" for i in range(250)]
    _chunked_query(client, ids, lambda b: "X " + values_clause(b), chunk=100, label="t")

    assert len(client.queries) == 3
    assert seen == [f"wd:Q{i}" for i in range(250)], "order and coverage must be exact"


def test_chunked_query_halves_on_truncation():
    """WDQS timeouts must shrink the chunk and retry, not sink the run."""
    calls = {"n": 0}

    def responder(query):
        calls["n"] += 1
        size = sum(1 for q in query.split() if q.startswith("wd:"))
        if size > 100:  # only chunks at or under 100 succeed
            raise QueryTruncated("timed out")
        return []

    client = _FakeClient(responder)
    ids = [f"Q{i}" for i in range(800)]
    _chunked_query(client, ids, lambda b: "X " + values_clause(b), chunk=800, label="t")

    ok = [
        sum(1 for q in query.split() if q.startswith("wd:"))
        for query in client.queries
        if sum(1 for q in query.split() if q.startswith("wd:")) <= 100
    ]
    assert max(ok) <= 100, "should have halved down to a size that succeeds"
    assert sum(ok) == 800, "every id must still be covered after halving"


def test_chunked_query_reraises_below_floor():
    """Below the floor a truncation is a real error, not an oversized chunk."""

    def responder(query):
        raise QueryTruncated("timed out")

    client = _FakeClient(responder)
    with pytest.raises(QueryTruncated):
        _chunked_query(
            client, [f"Q{i}" for i in range(CHUNK_FLOOR)],
            lambda b: "X " + values_clause(b), chunk=CHUNK_FLOOR, label="t",
        )


def test_humans_via_sparql_extracts_qids():
    client = _FakeClient(
        lambda q: [{"item": "http://www.wikidata.org/entity/Q7085"}]
    )
    assert humans_via_sparql(client, ["Q7085", "Q413"], chunk=50) == {"Q7085"}


def test_humans_via_sparql_omits_non_humans():
    """Query A returns only matches; absent QIDs are the non-humans."""
    client = _FakeClient(lambda q: [])
    assert humans_via_sparql(client, ["Q413"], chunk=50) == set()


def test_metadata_collapses_multiple_occupation_rows():
    """No GROUP_CONCAT, so one row per occupation arrives and must collapse."""
    base = "http://www.wikidata.org/entity/"
    rows = [
        {"item": base + "Q7085", "birthYear": "1885", "deathYear": "1962",
         "gender": base + "Q6581097", "occ": base + "Q169470"},
        {"item": base + "Q7085", "birthYear": "1885", "deathYear": "1962",
         "gender": base + "Q6581097", "occ": base + "Q170790"},
    ]
    client = _FakeClient(lambda q: rows)
    meta = metadata_via_sparql(client, ["Q7085"], chunk=50)

    assert set(meta) == {"Q7085"}
    assert meta["Q7085"]["birth_year"] == 1885
    assert meta["Q7085"]["death_year"] == 1962
    assert meta["Q7085"]["gender"] == "Q6581097"
    assert meta["Q7085"]["occupations"] == ["Q169470", "Q170790"], "sorted and deduped"


def test_metadata_tolerates_missing_optionals():
    client = _FakeClient(
        lambda q: [{"item": "http://www.wikidata.org/entity/Q1"}]
    )
    meta = metadata_via_sparql(client, ["Q1"], chunk=50)
    assert meta["Q1"] == {
        "birth_year": None, "death_year": None, "gender": None, "occupations": []
    }


def test_long_query_is_posted():
    """A VALUES clause of thousands of QIDs exceeds GET URL limits."""
    client = WikiClient("someone@example.org", delay=0)
    captured = {}

    def fake_request(method, url, **kwargs):
        captured["method"] = method
        captured["has_body"] = "data" in kwargs

        class R:
            status_code = 200
            text = '{"results":{"bindings":[]}}'

        return R()

    client._request = fake_request
    client.sparql("SELECT ?x WHERE { " + "wd:Q1 " * 1000 + "}")
    assert captured["method"] == "POST"
    assert captured["has_body"]


def test_short_query_uses_get():
    client = WikiClient("someone@example.org", delay=0)
    captured = {}

    def fake_get(url, **kwargs):
        captured["params"] = kwargs.get("params")

        class R:
            status_code = 200
            text = '{"results":{"bindings":[]}}'

        return R()

    client._get = fake_get
    client.sparql("SELECT ?x WHERE { ?x ?y ?z }")
    assert "query" in captured["params"]
