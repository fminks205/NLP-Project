"""Clean stage: raw Parsoid HTML -> documents.jsonl. Implements spec 0001 §Decision 3.

Wiki links are the reason this project stores HTML rather than plain text: a
link to ``./Niels_Bohr`` is an entity link a human wrote deliberately. They are
preserved here as **character offsets into the cleaned text**, so downstream
stages can treat the text as plain prose and still recover the links.

Whitespace is therefore collapsed *during* accumulation, never afterwards —
normalising the text after recording offsets would silently invalidate them.
"""

from __future__ import annotations

import re
import urllib.parse
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup, NavigableString, Tag

from ..manifest import read_jsonl, write_jsonl, write_manifest
from ..wiki import PERMALINK

LICENSE = "CC BY-SA 4.0"

#: Sections dropped wholesale, matched case-insensitively on the heading text.
DROP_SECTIONS = {
    "see also", "references", "external links", "further reading", "notes",
    "bibliography", "sources", "citations", "footnotes", "works cited",
    "publications", "selected publications", "external link", "notes and references",
}

#: Elements removed before any text is extracted.
DROP_SELECTORS = (
    "table", "figure", "figcaption", "style", "script", "link", "meta",
    "sup.mw-ref", "sup.reference", "[typeof~='mw:Extension/ref']",
    "[typeof~='mw:Extension/math']", "[typeof~='mw:Extension/references']",
    ".mw-editsection", ".navbox", ".metadata", ".hatnote", ".thumb",
    ".infobox", ".sidebar", ".ambox", ".mbox", "[role='navigation']",
    ".mw-empty-elt", ".shortdescription", ".noprint",
)

_WS = re.compile(r"\s+")


@dataclass
class _Accumulator:
    """Builds text and link spans together so offsets stay valid.

    Whitespace runs collapse to a single space as text arrives, which is what
    keeps ``text[start:end] == surface`` true.
    """

    parts: list[str] = field(default_factory=list)
    length: int = 0
    links: list[dict[str, Any]] = field(default_factory=list)
    _pending_space: bool = False

    def flush(self) -> None:
        """Emit any pending space.

        Must be called before recording a span's start offset, otherwise the
        span swallows the space that precedes it — the offsets stay internally
        consistent (``text[start:end] == surface`` still holds) while both are
        wrong, which is exactly the bug a round-trip check cannot see.
        """
        if self._pending_space and self.length:
            self.parts.append(" ")
            self.length += 1
        self._pending_space = False

    def add(self, raw: str) -> None:
        text = _WS.sub(" ", raw)
        if not text:
            return
        if text.startswith(" "):
            self._pending_space = True
            text = text.lstrip()
            if not text:
                return
        trailing = text.endswith(" ")
        text = text.rstrip()
        self.flush()
        self._pending_space = trailing
        self.parts.append(text)
        self.length += len(text)

    def text(self) -> str:
        return "".join(self.parts)


def _title_from_href(href: str) -> str | None:
    """Turn a Parsoid wikilink href into an article title.

    ``./Niels_Bohr`` -> ``Niels Bohr``. Fragment-only and external links return
    None so they are not recorded as entity links.

    Red links — targets with no article yet — arrive as
    ``./Wolfgang_Riezler?action=edit&redlink=1``. The query string must be stripped
    or the title is unusable: it will match nothing in Wikidata and shows up as a
    mangled pseudo-title. The link still names a real person, so the title is kept
    and the caller decides what to do with an unresolvable one.
    """
    if not href or not href.startswith("./"):
        return None
    target = href[2:].split("#", 1)[0].split("?", 1)[0]
    if not target:
        return None
    return urllib.parse.unquote(target).replace("_", " ")


def _walk(node: Tag, acc: _Accumulator) -> None:
    """Accumulate text from node, recording wikilink spans as they close."""
    for child in node.children:
        if isinstance(child, NavigableString):
            acc.add(str(child))
        elif isinstance(child, Tag):
            rel = child.get("rel") or []
            is_wikilink = child.name == "a" and "mw:WikiLink" in rel
            if is_wikilink:
                acc.flush()  # so the span starts at content, not at whitespace
                start = acc.length
                _walk(child, acc)
                end = acc.length
                title = _title_from_href(child.get("href", ""))
                if title and end > start:
                    # Whitespace can also arrive from *inside* the anchor
                    # (e.g. <a><span> Kaiser ...</span></a>), which lands after
                    # `start` was taken. Tighten the span onto real content.
                    text = acc.text()
                    raw = text[start:end]
                    start += len(raw) - len(raw.lstrip())
                    end -= len(raw) - len(raw.rstrip())
                    if end > start:
                        acc.links.append(
                            {
                                "start": start,
                                "end": end,
                                "surface": text[start:end],
                                "target_title": title,
                            }
                        )
            else:
                _walk(child, acc)


def _heading_of(section: Tag) -> str:
    """Heading text of a Parsoid section, or '' for the lead."""
    for level in ("h2", "h3", "h4", "h5", "h6"):
        heading = section.find(level, recursive=False)
        if heading:
            return heading.get_text(" ", strip=True)
    return ""


def parse_html(html: bytes | str) -> list[dict[str, Any]]:
    """Parse Parsoid HTML into cleaned sections with link spans."""
    soup = BeautifulSoup(html, "lxml")

    for selector in DROP_SELECTORS:
        for element in soup.select(selector):
            element.decompose()

    body = soup.body or soup
    sections = [s for s in body.find_all("section") if s.find_parent("section") is None]
    if not sections:  # older or unsectioned output
        sections = [body]

    out: list[dict[str, Any]] = []
    for section in sections:
        heading = _heading_of(section)
        if heading.strip().lower() in DROP_SECTIONS:
            continue

        paragraphs: list[str] = []
        links: list[dict[str, Any]] = []
        offset = 0
        for para in section.find_all("p", recursive=True):
            acc = _Accumulator()
            _walk(para, acc)
            text = acc.text().strip()
            if not text:
                continue
            # Re-base this paragraph's link offsets onto the section text.
            lead = len(acc.text()) - len(acc.text().lstrip())
            for link in acc.links:
                link["start"] += offset - lead
                link["end"] += offset - lead
            links.extend(l for l in acc.links if l["start"] >= offset)
            paragraphs.append(text)
            offset += len(text) + 2  # "\n\n"

        if not paragraphs:
            continue
        out.append({"heading": heading, "text": "\n\n".join(paragraphs), "links": links})
    return out


def verify_offsets(sections: list[dict[str, Any]]) -> list[str]:
    """Return a list of link-span problems. Empty means the spans are sound.

    Checks two things, not one. Round-tripping ``text[start:end] == surface``
    only proves the offsets agree with the recorded surface — both can be
    wrong together. The whitespace check catches spans that are internally
    consistent but shifted off the actual anchor text.
    """
    problems = []
    for index, section in enumerate(sections):
        text = section["text"]
        for link in section["links"]:
            actual = text[link["start"] : link["end"]]
            if actual != link["surface"]:
                problems.append(
                    f"section {index} ({section['heading']!r}): "
                    f"offset mismatch, expected {link['surface']!r}, got {actual!r}"
                )
            elif actual != actual.strip():
                problems.append(
                    f"section {index} ({section['heading']!r}): "
                    f"span includes surrounding whitespace: {actual!r}"
                )
    return problems


def run(
    raw_dir: Path,
    out_path: Path,
    *,
    seed_path: Path,
    limit: int | None = None,
    strict: bool = True,
) -> dict[str, Any]:
    """Clean every fetched article into documents.jsonl."""
    people = {p["qid"]: p for p in read_jsonl(seed_path)}
    log_path = raw_dir / "fetch_log.jsonl"
    revisions: dict[str, dict] = {}
    if log_path.exists():
        for entry in read_jsonl(log_path):
            if entry.get("status") == "ok":
                revisions[entry["qid"]] = entry

    html_files = sorted((raw_dir / "html").glob("*.html"))
    if limit:
        html_files = html_files[:limit]

    documents: list[dict[str, Any]] = []
    counts = {"documents": 0, "sections": 0, "links": 0, "offset_errors": 0, "empty": 0}

    for path in html_files:
        qid = path.stem
        person = people.get(qid, {})
        entry = revisions.get(qid, {})
        sections = parse_html(path.read_bytes())

        problems = verify_offsets(sections)
        if problems:
            counts["offset_errors"] += len(problems)
            if strict:
                raise ValueError(f"{qid}: link offsets do not round-trip: {problems[:3]}")

        if not sections:
            counts["empty"] += 1
            continue

        revision_id = entry.get("revision_id")
        documents.append(
            {
                "doc_id": qid,
                "title": person.get("title") or qid,
                "qid": qid,
                "revision_id": revision_id,
                "url": PERMALINK.format(revid=revision_id) if revision_id else None,
                "fetched_at": entry.get("fetched_at") or datetime.now(UTC).isoformat(),
                "license": LICENSE,
                "sections": sections,
            }
        )
        counts["documents"] += 1
        counts["sections"] += len(sections)
        counts["links"] += sum(len(s["links"]) for s in sections)

    write_jsonl(out_path, documents)
    write_manifest(
        out_path.parent,
        stage="clean",
        config={"raw_dir": str(raw_dir), "limit": limit, "strict": strict},
        counts=counts,
        inputs={"seed": seed_path},
    )
    return counts
