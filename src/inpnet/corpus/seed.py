"""Seed stage: Wikidata SPARQL -> data/raw/seed.jsonl. Implements spec 0001 §Decision 1.

Two behaviours here exist because of things measured on 2026-08-12, not because
of taste:

* The query must not use ``SERVICE wikibase:label``. With it, the full physicist
  query takes 67s and is truncated by the WDQS timeout; without it, 9.6s and
  clean. Display names are derived from the article title instead, which is
  the same string a reader would see.
* ``OPTIONAL`` clauses multiply rows — 18,178 rows for 15,158 distinct people,
  because Wikidata holds several birth dates or genders for some. Dedup is
  mandatory, not defensive.
"""

from __future__ import annotations

import urllib.parse
from pathlib import Path
from typing import Any

from ..manifest import write_jsonl, write_manifest
from ..wiki import WikiClient


def title_from_article_url(url: str) -> str:
    """Turn an enwiki URL into its display title.

    >>> title_from_article_url("https://en.wikipedia.org/wiki/Niels_Bohr")
    'Niels Bohr'
    """
    slug = url.rsplit("/", 1)[-1]
    return urllib.parse.unquote(slug).replace("_", " ")


def qid_from_entity_url(url: str) -> str:
    """Extract the QID from a Wikidata entity URL."""
    return url.rsplit("/", 1)[-1]


def collapse_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    """Collapse duplicated SPARQL rows into one record per person.

    Wikidata may hold multiple values for birth date, death date or gender. The
    SPARQL OPTIONALs then emit their cross product. We keep the first value seen
    for each scalar field and record how many people were affected so the
    manifest can report it.
    """
    people: dict[str, dict[str, Any]] = {}
    for row in rows:
        qid = qid_from_entity_url(row["person"])
        record = people.get(qid)
        if record is None:
            record = {
                "qid": qid,
                "title": title_from_article_url(row["article"]),
                "birth_year": _as_int(row.get("birthYear")),
                "death_year": _as_int(row.get("deathYear")),
                "gender": qid_from_entity_url(row["gender"]) if row.get("gender") else None,
                "_variants": 0,
            }
            people[qid] = record
        else:
            record["_variants"] += 1
            # Fill gaps a later row may cover, but never overwrite a set value.
            if record["birth_year"] is None:
                record["birth_year"] = _as_int(row.get("birthYear"))
            if record["death_year"] is None:
                record["death_year"] = _as_int(row.get("deathYear"))
            if record["gender"] is None and row.get("gender"):
                record["gender"] = qid_from_entity_url(row["gender"])

    return sorted(people.values(), key=lambda r: r["qid"])


def _as_int(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def run(
    query_path: Path,
    out_path: Path,
    *,
    contact: str,
    delay: float = 1.0,
) -> dict[str, Any]:
    """Run the seed query and write seed.jsonl. Returns summary counts."""
    query = query_path.read_text(encoding="utf-8")
    client = WikiClient(contact, delay=delay)

    rows = client.sparql(query)
    people = collapse_rows(rows)

    duplicated = sum(1 for p in people if p["_variants"])
    for person in people:
        del person["_variants"]

    write_jsonl(out_path, people)

    counts = {
        "sparql_rows": len(rows),
        "distinct_people": len(people),
        "people_with_duplicate_rows": duplicated,
        "with_birth_year": sum(1 for p in people if p["birth_year"] is not None),
        "with_death_year": sum(1 for p in people if p["death_year"] is not None),
        "with_gender": sum(1 for p in people if p["gender"]),
    }
    write_manifest(
        out_path.parent,
        stage="seed",
        spec="0001-corpus-acquisition",
        config={"query_path": str(query_path), "endpoint": "WDQS"},
        counts=counts,
        extra={"query": query},
    )
    return counts
