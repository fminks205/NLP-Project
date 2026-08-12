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
REST_HTML = "https://en.wikipedia.org/api/rest_v1/page/html/"
REST_BARE = "https://en.wikipedia.org/w/rest.php/v1/page/{title}/bare"
PERMALINK = "https://en.wikipedia.org/w/index.php?oldid={revid}"

#: The Action API accepts at most 50 titles per request for anonymous clients.
TITLES_PER_REQUEST = 50

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
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            self._throttle()
            try:
                resp = self.session.get(url, timeout=self.timeout, **kwargs)
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

    def sparql(self, query: str) -> list[dict[str, str]]:
        """Run a SPARQL query, returning one flat dict of bindings per row.

        Raises:
            QueryTruncated: if WDQS timed out mid-stream. See _TRUNCATION_MARKERS —
                this failure arrives as HTTP 200 and must be detected from the body.
        """
        resp = self._get(
            WDQS_ENDPOINT,
            params={"query": query},
            headers={"Accept": "application/sparql-results+json"},
        )
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


def _revision_from_etag(etag: str) -> int | None:
    """Parse the revision id out of a Parsoid ETag, or None if absent."""
    if '"' not in etag:
        return None
    head = etag.split('"')[1].split("/")[0]
    return int(head) if head.isdigit() else None
