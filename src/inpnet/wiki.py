"""HTTP access to the Wikimedia APIs. Implements spec 0001 §Decision 2.

Etiquette is a requirement of that spec, not an optimisation: every request
carries a descriptive User-Agent with a contact address, requests are serial
with a floor on the inter-request delay, and retries back off and honour
Retry-After.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import requests

WDQS_ENDPOINT = "https://query.wikidata.org/sparql"
ACTION_API = "https://en.wikipedia.org/w/api.php"
WIKIDATA_API = "https://www.wikidata.org/w/api.php"
REST_HTML = "https://en.wikipedia.org/api/rest_v1/page/html/"
REST_BARE = "https://en.wikipedia.org/w/rest.php/v1/page/{title}/bare"
PERMALINK = "https://en.wikipedia.org/w/index.php?oldid={revid}"

#: The Action API accepts at most 50 titles per request for anonymous clients.
TITLES_PER_REQUEST = 50

#: Queries longer than this are POSTed rather than sent in a URL.
_POST_THRESHOLD = 2000

#: WDQS streams results and, when a query exceeds its server-side timeout,
#: appends a Java stack trace to the already-emitted JSON *while still
#: returning HTTP 200*. A truncated seed list would otherwise pass silently,
#: so these markers are checked explicitly. Observed 2026-08-12.
_TRUNCATION_MARKERS = (
    "SPARQL-QUERY:",
    "java.util.concurrent",
    "TimeoutException",
    "QueryTimeoutException",
)


class WikiError(RuntimeError):
    """Any failure talking to a Wikimedia API."""


class QueryTruncated(WikiError):
    """WDQS returned HTTP 200 but the result stream was cut short."""


@dataclass(frozen=True)
class Page:
    """One fetched article, with the revision it was pinned to."""

    title: str
    revision_id: int
    html: bytes

    @property
    def url(self) -> str:
        """Permanent link to the exact revision this content came from."""
        return PERMALINK.format(revid=self.revision_id)


class WikiClient:
    """Polite, serial client for WDQS and the Wikipedia APIs.

    Args:
        contact: Email or URL identifying the operator. Wikimedia's User-Agent
            policy requires a real contact point; there is no default.
        delay: Minimum seconds between requests.
        timeout: Per-request socket timeout in seconds.
        max_retries: Attempts per request before giving up.
    """

    def __init__(
        self,
        contact: str,
        *,
        delay: float = 1.0,
        timeout: float = 90.0,
        max_retries: int = 4,
        session: requests.Session | None = None,
    ) -> None:
        if not contact or "@" not in contact and "://" not in contact:
            raise ValueError(
                "contact must be an email address or URL — Wikimedia's "
                "User-Agent policy requires an identifiable contact point"
            )
        self.contact = contact
        self.delay = delay
        self.timeout = timeout
        self.max_retries = max_retries
        self.session = session or requests.Session()
        self.session.headers["User-Agent"] = (
            f"inpnet/0.1 (FH-SWF AKI NLP research project; {contact}) python-requests"
        )
        self._last_request = 0.0

    # -- plumbing ---------------------------------------------------------

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_request
        if elapsed < self.delay:
            time.sleep(self.delay - elapsed)
        self._last_request = time.monotonic()

    def _get(self, url: str, **kwargs) -> requests.Response:
        """GET with throttling and backoff. Raises WikiError when out of retries."""
        return self._request("GET", url, **kwargs)

    def _request(self, method: str, url: str, **kwargs) -> requests.Response:
        """HTTP with throttling and backoff. Raises WikiError when out of retries."""
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            self._throttle()
            try:
                resp = self.session.request(method, url, timeout=self.timeout, **kwargs)
            except requests.RequestException as exc:
                last_error = exc
            else:
                if resp.status_code == 200:
                    return resp
                if resp.status_code in (429, 502, 503, 504):
                    retry_after = resp.headers.get("Retry-After")
                    wait = (
                        float(retry_after)
                        if retry_after and retry_after.isdigit()
                        else 2.0**attempt
                    )
                    last_error = WikiError(f"HTTP {resp.status_code} from {url}")
                    time.sleep(wait)
                    continue
                if resp.status_code == 404:
                    raise WikiError(f"HTTP 404: {url}")
                last_error = WikiError(f"HTTP {resp.status_code} from {url}")
            time.sleep(2.0**attempt)
        raise WikiError(f"giving up on {url} after {self.max_retries} attempts") from last_error

    # -- Wikidata ---------------------------------------------------------

    def sparql(self, query: str, *, method: str | None = None) -> list[dict[str, str]]:
        """Run a SPARQL query, returning one flat dict of bindings per row.

        Long queries — a `VALUES` clause of thousands of QIDs runs to tens of
        kilobytes — exceed practical GET URL limits, so anything past
        `_POST_THRESHOLD` is sent as a form-encoded POST. Verified working against
        WDQS on 2026-08-25.

        Raises:
            QueryTruncated: if WDQS timed out mid-stream. See _TRUNCATION_MARKERS —
                this failure arrives as HTTP 200 and must be detected from the body.
        """
        if method is None:
            method = "POST" if len(query) > _POST_THRESHOLD else "GET"
        headers = {"Accept": "application/sparql-results+json"}
        if method == "POST":
            resp = self._request(
                "POST", WDQS_ENDPOINT, data={"query": query}, headers=headers
            )
        else:
            resp = self._get(WDQS_ENDPOINT, params={"query": query}, headers=headers)
        text = resp.text
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            tail = text[-400:]
            if any(m in text for m in _TRUNCATION_MARKERS):
                raise QueryTruncated(
                    "WDQS timed out and returned a partial result stream "
                    f"({len(text):,} chars). Simplify or paginate the query. Tail: {tail!r}"
                ) from exc
            raise WikiError(f"unparseable WDQS response, tail: {tail!r}") from exc

        # A stack trace can also land *inside* otherwise-valid JSON.
        if any(m in text for m in _TRUNCATION_MARKERS):
            raise QueryTruncated("WDQS response contains a server error trace")

        return [
            {var: binding["value"] for var, binding in row.items()}
            for row in payload["results"]["bindings"]
        ]

    # -- Wikipedia --------------------------------------------------------

    def page_info(self, titles: Sequence[str]) -> Iterator[dict]:
        """Yield page metadata, including wikitext `length`, in batches of 50.

        Used by the estimate stage: `length` is the exact wikitext byte count,
        so total corpus size is measured rather than guessed.
        """
        for start in range(0, len(titles), TITLES_PER_REQUEST):
            batch = titles[start : start + TITLES_PER_REQUEST]
            resp = self._get(
                ACTION_API,
                params={
                    "action": "query",
                    "prop": "info",
                    "redirects": "1",
                    "titles": "|".join(batch),
                    "format": "json",
                    "formatversion": "2",
                },
            )
            data = resp.json().get("query", {})
            redirects = {r["from"]: r["to"] for r in data.get("redirects", [])}
            for page in data.get("pages", []):
                yield {
                    "title": page["title"],
                    "length": page.get("length"),
                    "lastrevid": page.get("lastrevid"),
                    "missing": bool(page.get("missing")),
                    "redirected_from": next(
                        (src for src, dst in redirects.items() if dst == page["title"]),
                        None,
                    ),
                }

    def page_props(self, titles: Sequence[str]) -> Iterator[dict]:
        """Yield title -> Wikidata QID, in batches of 50. Implements spec 0002 §Decision 2.

        `redirects=1` resolves redirects, so two requested titles may map to the same
        QID. That is correct — they are the same entity.
        """
        for start in range(0, len(titles), TITLES_PER_REQUEST):
            batch = list(titles[start : start + TITLES_PER_REQUEST])
            resp = self._get(
                ACTION_API,
                params={
                    "action": "query",
                    "prop": "pageprops",
                    "ppprop": "wikibase_item",
                    "redirects": "1",
                    "titles": "|".join(batch),
                    "format": "json",
                    "formatversion": "2",
                },
            )
            data = resp.json().get("query", {})
            # Map every requested title through normalisation and redirects to the
            # title the API actually returned, so nothing is silently dropped.
            alias: dict[str, str] = {}
            for hop in ("normalized", "redirects"):
                for entry in data.get(hop, []):
                    alias[entry["from"]] = entry["to"]

            resolved = {
                page["title"]: page.get("pageprops", {}).get("wikibase_item")
                for page in data.get("pages", [])
            }
            for title in batch:
                final = title
                for _ in range(4):  # normalise -> redirect chains are short
                    if final in alias:
                        final = alias[final]
                    else:
                        break
                yield {
                    "title": title,
                    "resolved_title": final,
                    "qid": resolved.get(final),
                }

    def wikidata_entities(self, qids: Sequence[str]) -> Iterator[dict]:
        """Yield Wikidata claims for each QID, in batches of 50.

        Returns everything spec 0002 needs from one call: whether the item is a human
        (`P31 = Q5`), plus birth year, death year, gender and occupations.
        """
        for start in range(0, len(qids), TITLES_PER_REQUEST):
            batch = list(qids[start : start + TITLES_PER_REQUEST])
            resp = self._get(
                WIKIDATA_API,
                params={
                    "action": "wbgetentities",
                    "ids": "|".join(batch),
                    "props": "claims",
                    "format": "json",
                },
            )
            entities = resp.json().get("entities", {})
            for qid in batch:
                entity = entities.get(qid)
                if entity is None or "missing" in entity:
                    yield {"qid": qid, "is_human": False, "missing": True}
                    continue
                claims = entity.get("claims", {})
                yield {
                    "qid": qid,
                    "missing": False,
                    "is_human": "Q5" in _claim_ids(claims, "P31"),
                    "birth_year": _claim_year(claims, "P569"),
                    "death_year": _claim_year(claims, "P570"),
                    "gender": next(iter(_claim_ids(claims, "P21")), None),
                    "occupations": sorted(_claim_ids(claims, "P106")),
                }

    def page_html(self, title: str) -> Page:
        """Fetch Parsoid HTML and the revision id it was rendered from.

        The revision id comes from the response ETag —
        ``W/"1366956682/<uuid>/view/html"`` — so pinning costs no extra request.
        Verified against Niels_Bohr on 2026-08-12.
        """
        resp = self._get(REST_HTML + requests.utils.quote(title.replace(" ", "_"), safe=""))
        revision_id = _revision_from_etag(resp.headers.get("ETag", ""))
        if revision_id is None:
            raise WikiError(f"no revision id in ETag for {title!r}: {resp.headers.get('ETag')!r}")
        return Page(title=title, revision_id=revision_id, html=resp.content)


def _best_rank(claims: dict, prop: str) -> list[dict]:
    """Claims for `prop` at best rank — preferred if any exist, else normal.

    Deprecated claims are never returned. Ignoring rank is not a detail: Q17021508
    is a *helicopter* carrying a deprecated `P31 = Q5`, and reading every claim
    regardless of rank classified it as a person. SPARQL's `wdt:` prefix applies
    exactly this rule, so the two paths now agree.
    """
    candidates = [c for c in claims.get(prop, []) if c.get("rank") != "deprecated"]
    preferred = [c for c in candidates if c.get("rank") == "preferred"]
    return preferred or candidates


def _claim_ids(claims: dict, prop: str) -> set[str]:
    """Entity ids asserted by `prop` at best rank, ignoring novalue/somevalue snaks."""
    out = set()
    for claim in _best_rank(claims, prop):
        value = claim.get("mainsnak", {}).get("datavalue", {}).get("value")
        if isinstance(value, dict) and "id" in value:
            out.add(value["id"])
    return out


def _claim_year(claims: dict, prop: str) -> int | None:
    """Year from the first time-valued claim for `prop`.

    Wikidata times look like ``+1885-10-07T00:00:00Z``, and BCE dates carry a leading
    ``-``, so the sign is kept rather than stripped.

    Note this returns the year **as stored**, which for older records is often the
    Julian calendar. WDQS's ``YEAR()`` converts to Gregorian first, so the two can
    differ by a year around a New Year boundary — Q1232515 is stored as 29 Dec 1907
    Julian, which is 11 Jan 1908 Gregorian. SPARQL is the primary path and gives the
    Gregorian answer.
    """
    for claim in _best_rank(claims, prop):
        value = claim.get("mainsnak", {}).get("datavalue", {}).get("value")
        if isinstance(value, dict) and "time" in value:
            stamp = value["time"]
            sign, digits = (-1, stamp[1:]) if stamp.startswith("-") else (1, stamp.lstrip("+"))
            year = digits.split("-", 1)[0]
            if year.isdigit():
                return sign * int(year)
    return None


def _revision_from_etag(etag: str) -> int | None:
    """Parse the revision id out of a Parsoid ETag, or None if absent."""
    if '"' not in etag:
        return None
    head = etag.split('"')[1].split("/")[0]
    return int(head) if head.isdigit() else None
