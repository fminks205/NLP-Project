"""Tests for the Wikimedia client. Implements spec 0001 §Decision 2."""

import json

import pytest

from inpnet.wiki import QueryTruncated, WikiClient, WikiError, _revision_from_etag


def test_revision_parsed_from_parsoid_etag():
    """Verified against Niels_Bohr on 2026-08-12 — this is how pinning stays free."""
    etag = 'W/"1366956682/4258f96d-964c-11f1-919f-ffd90e6059dd/view/html"'
    assert _revision_from_etag(etag) == 1366956682


@pytest.mark.parametrize("etag", ["", "no-quotes", 'W/"notanumber/x"'])
def test_revision_from_bad_etag_is_none(etag):
    assert _revision_from_etag(etag) is None


def test_contact_is_required():
    with pytest.raises(ValueError, match="contact"):
        WikiClient("")


def test_contact_must_look_like_a_contact():
    with pytest.raises(ValueError, match="contact"):
        WikiClient("anonymous")


def test_user_agent_carries_contact():
    client = WikiClient("someone@example.org")
    assert "someone@example.org" in client.session.headers["User-Agent"]


class _FakeResponse:
    def __init__(self, text):
        self.text = text
        self.status_code = 200


def _client_returning(text, monkeypatch):
    client = WikiClient("someone@example.org", delay=0)
    monkeypatch.setattr(client, "_get", lambda *a, **k: _FakeResponse(text))
    return client


def test_sparql_parses_bindings(monkeypatch):
    payload = json.dumps(
        {"results": {"bindings": [{"person": {"value": "Q1"}, "n": {"value": "2"}}]}}
    )
    client = _client_returning(payload, monkeypatch)
    assert client.sparql("SELECT ...") == [{"person": "Q1", "n": "2"}]


def test_sparql_detects_wdqs_timeout_truncation(monkeypatch):
    """WDQS returns HTTP 200 with a partial stream plus a stack trace on timeout.

    Observed 2026-08-12: the physicist query with SERVICE wikibase:label ran 67s
    and came back truncated. Silently accepting that yields an incomplete corpus,
    so it must raise.
    """
    truncated = (
        '{"head":{"vars":["person"]},"results":{"bindings":[{"person":{"type":"l'
        "SPARQL-QUERY: queryStr=SELECT ...\n"
        "java.util.concurrent.TimeoutException\n"
        "\tat java.lang.Thread.run(Thread.java:750)\n"
    )
    client = _client_returning(truncated, monkeypatch)
    with pytest.raises(QueryTruncated, match="partial result stream"):
        client.sparql("SELECT ...")


def test_sparql_raises_on_unparseable_response(monkeypatch):
    client = _client_returning("<html>gateway error</html>", monkeypatch)
    with pytest.raises(WikiError):
        client.sparql("SELECT ...")
