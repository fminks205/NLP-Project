"""Resolve wiki-link targets to Wikidata and flag the humans. Spec 0002 §Decision 2.

Two passes, each cached to its own file so a long run can be interrupted and resumed:

    pass 1   title -> QID      (en.wikipedia.org, prop=pageprops, 50 per request)
    pass 2   QID   -> claims   (WDQS, chunked SPARQL VALUES)

Pass 2 is SPARQL rather than `wbgetentities` because a `VALUES` clause carries thousands
of QIDs per request: ~30 requests instead of ~1,954 for this corpus.

It runs as two queries, not one. Query A asks only "which of these are human?" — cheap,
no OPTIONALs. Query B fetches metadata for the humans alone, which is ~13% of the input,
so the expensive query does an eighth of the work. Neither uses `GROUP_CONCAT` or
`SERVICE wikibase:label`: both are what pushed the spec 0001 seed query past the WDQS
timeout.
"""

from __future__ import annotations

import gzip
import json
import shutil
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Iterable

from ..manifest import read_jsonl, write_jsonl, write_manifest
from ..wiki import QueryTruncated, WikiClient

#: QIDs per request. Query B carries OPTIONALs, so it gets smaller chunks.
CHUNK_HUMANS = 2000
CHUNK_METADATA = 1000
#: Below this, a truncation is a real failure rather than an oversized chunk.
#: Kept low so a genuinely slow region degrades gracefully instead of failing the run;
#: if WDQS cannot answer for 25 explicit QIDs, retrying smaller will not help.
CHUNK_FLOOR = 25


def collect_link_targets(documents_path: Path) -> Counter[str]:
    """Count link-target titles across the corpus, most frequent first."""
    counts: Counter[str] = Counter()
    with documents_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            for section in json.loads(line)["sections"]:
                for link in section["links"]:
                    counts[link["target_title"]] += 1
    return counts


def values_clause(qids: Iterable[str]) -> str:
    """Render QIDs as a SPARQL VALUES body: ``wd:Q1 wd:Q2 ...``"""
    return " ".join(f"wd:{q}" for q in qids)


def _chunked_query(
    client: WikiClient,
    ids: list[str],
    build: Callable[[list[str]], str],
    chunk: int,
    label: str,
) -> list[dict[str, str]]:
    """Run `build` over `ids` in chunks, halving any chunk WDQS times out on.

    WDQS signals a timeout as HTTP 200 with a truncated body, which `sparql()` turns
    into QueryTruncated. Halving makes an oversized chunk self-correcting instead of
    fatal, so a single slow region cannot sink the whole run.
    """
    rows: list[dict[str, str]] = []
    queue = [ids[i : i + chunk] for i in range(0, len(ids), chunk)]
    done = 0
    while queue:
        batch = queue.pop(0)
        try:
            rows.extend(client.sparql(build(batch)))
        except QueryTruncated:
            if len(batch) <= CHUNK_FLOOR:
                raise
            middle = len(batch) // 2
            print(f"  {label}: chunk of {len(batch)} timed out, halving", flush=True)
            queue[:0] = [batch[:middle], batch[middle:]]
            continue
        done += len(batch)
        print(f"  {label}: {done:,}/{len(ids):,}", flush=True)
    return rows


def humans_via_sparql(client: WikiClient, qids: list[str], chunk: int = CHUNK_HUMANS) -> set[str]:
    """Return the subset of `qids` that are instances of human (P31 = Q5)."""

    def build(batch: list[str]) -> str:
        return (
            "SELECT ?item WHERE { VALUES ?item { "
            + values_clause(batch)
            + " } ?item wdt:P31 wd:Q5 }"
        )

    rows = _chunked_query(client, qids, build, chunk, "humans")
    return {row["item"].rsplit("/", 1)[-1] for row in rows}


def metadata_via_sparql(
    client: WikiClient, qids: list[str], chunk: int = CHUNK_METADATA
) -> dict[str, dict]:
    """Fetch birth/death year, gender and occupations for the given QIDs.

    Occupations arrive as one row per (item, occupation) — no GROUP_CONCAT — and are
    collapsed here, the same way `corpus/seed.py` collapses duplicated OPTIONAL rows.
    """

    def build(batch: list[str]) -> str:
        return (
            "SELECT ?item ?birthYear ?deathYear ?gender ?occ WHERE { VALUES ?item { "
            + values_clause(batch)
            + " } "
            "OPTIONAL { ?item wdt:P569 ?b . BIND(YEAR(?b) AS ?birthYear) } "
            "OPTIONAL { ?item wdt:P570 ?d . BIND(YEAR(?d) AS ?deathYear) } "
            "OPTIONAL { ?item wdt:P21  ?gender } "
            "OPTIONAL { ?item wdt:P106 ?occ } }"
        )

    rows = _chunked_query(client, qids, build, chunk, "metadata")

    out: dict[str, dict] = {}
    occupations: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        qid = row["item"].rsplit("/", 1)[-1]
        record = out.setdefault(
            qid, {"birth_year": None, "death_year": None, "gender": None}
        )
        if record["birth_year"] is None:
            record["birth_year"] = _as_int(row.get("birthYear"))
        if record["death_year"] is None:
            record["death_year"] = _as_int(row.get("deathYear"))
        if record["gender"] is None and row.get("gender"):
            record["gender"] = row["gender"].rsplit("/", 1)[-1]
        if row.get("occ"):
            occupations[qid].add(row["occ"].rsplit("/", 1)[-1])
    for qid, record in out.items():
        record["occupations"] = sorted(occupations[qid])
    return out


def _as_int(value: str | None) -> int | None:
    try:
        return int(value) if value is not None else None
    except ValueError:
        return None


def _load_cache(path: Path, key: str) -> dict[str, dict]:
    if not path.exists():
        return {}
    return {row[key]: row for row in read_jsonl(path)}


def _append(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def run(
    documents_path: Path,
    out_dir: Path,
    *,
    contact: str,
    delay: float = 0.2,
    limit: int | None = None,
) -> dict[str, Any]:
    """Resolve every distinct link target; write the joined, dated cache."""
    started = datetime.now(UTC).isoformat()
    targets = collect_link_targets(documents_path)
    titles = [t for t, _ in targets.most_common()]
    if limit:
        titles = titles[:limit]

    client = WikiClient(contact, delay=delay)
    title_path = out_dir / "title_qid.jsonl"
    claim_path = out_dir / "qid_claims.jsonl"

    # -- pass 1: title -> QID (Action API, 50 per request) ----------------
    title_cache = _load_cache(title_path, "title")
    todo = [t for t in titles if t not in title_cache]
    print(f"pass 1: {len(todo):,} titles to resolve ({len(title_cache):,} cached)", flush=True)

    buffer: list[dict] = []
    for index, row in enumerate(client.page_props(todo), start=1):
        row["fetched_at"] = datetime.now(UTC).isoformat()
        title_cache[row["title"]] = row
        buffer.append(row)
        if len(buffer) >= 500:
            _append(title_path, buffer)
            buffer = []
        if index % 2500 == 0:
            print(f"  {index:,}/{len(todo):,}", flush=True)
    _append(title_path, buffer)

    # -- pass 2: QID -> claims (WDQS, chunked VALUES) ---------------------
    claim_cache = _load_cache(claim_path, "qid")
    wanted = {r["qid"] for r in title_cache.values() if r.get("qid")}
    todo_qids = sorted(wanted - set(claim_cache))
    print(f"pass 2: {len(todo_qids):,} QIDs to inspect ({len(claim_cache):,} cached)", flush=True)

    if todo_qids:
        humans = humans_via_sparql(client, todo_qids)
        print(f"  {len(humans):,} of {len(todo_qids):,} are human", flush=True)
        metadata = metadata_via_sparql(client, sorted(humans)) if humans else {}

        stamp = datetime.now(UTC).isoformat()
        fresh = []
        for qid in todo_qids:
            meta = metadata.get(qid, {})
            fresh.append(
                {
                    "qid": qid,
                    "missing": False,
                    "is_human": qid in humans,
                    "birth_year": meta.get("birth_year"),
                    "death_year": meta.get("death_year"),
                    "gender": meta.get("gender"),
                    "occupations": meta.get("occupations", []),
                    "fetched_at": stamp,
                    "method": "sparql",
                }
            )
        _append(claim_path, fresh)
        claim_cache.update({r["qid"]: r for r in fresh})

    # -- join -------------------------------------------------------------
    joined = []
    for title in titles:
        entry = title_cache.get(title)
        if not entry:
            continue
        qid = entry.get("qid")
        claims = claim_cache.get(qid or "")
        # Three distinct states, which must not collapse into one boolean:
        #   no QID at all   -> resolved; the target has no Wikidata item, so it is
        #                      definitively not an identifiable person (red links,
        #                      pages Wikidata does not cover). 5,768 of these.
        #   QID + claims    -> resolved; is_human is the looked-up answer.
        #   QID, no claims  -> NOT resolved; unknown. Collapsing this to False is what
        #                      let an incomplete cache silently drop real people.
        resolved = qid is None or claims is not None
        if qid is None:
            is_human: bool | None = False
        elif claims is not None:
            is_human = bool(claims["is_human"])
        else:
            is_human = None
        joined.append(
            {
                "title": title,
                "resolved_title": entry.get("resolved_title"),
                "qid": qid,
                "claims_resolved": resolved,
                "no_wikidata_item": qid is None,
                "is_human": is_human,
                "birth_year": (claims or {}).get("birth_year"),
                "death_year": (claims or {}).get("death_year"),
                "gender": (claims or {}).get("gender"),
                "occupations": (claims or {}).get("occupations") or [],
                "link_instances": targets[title],
                "fetched_at": (claims or {}).get("fetched_at") or entry.get("fetched_at"),
            }
        )

    cache_path = out_dir / "wikidata_cache.jsonl"
    write_jsonl(cache_path, joined)
    # Archived alongside so the downstream pipeline reproduces with no network.
    with cache_path.open("rb") as src, gzip.open(f"{cache_path}.gz", "wb", compresslevel=9) as dst:
        shutil.copyfileobj(src, dst)

    human_titles = [r for r in joined if r["is_human"]]
    human_instances = sum(r["link_instances"] for r in human_titles)
    total_instances = sum(targets[t] for t in titles)
    counts = {
        "distinct_targets": len(titles),
        "resolved_to_qid": sum(1 for r in joined if r["qid"]),
        "claims_unresolved": sum(1 for r in joined if not r["claims_resolved"]),
        "no_wikidata_item": sum(1 for r in joined if r["no_wikidata_item"]),
        "human_targets": len(human_titles),
        "total_link_instances": total_instances,
        "human_link_instances": human_instances,
        "human_instance_rate": round(human_instances / total_instances, 4)
        if total_instances
        else 0.0,
    }
    write_manifest(
        out_dir,
        stage="resolve",
        spec="0002-entity-mention-layer",
        config={
            "documents": str(documents_path),
            "limit": limit,
            "chunk_humans": CHUNK_HUMANS,
            "chunk_metadata": CHUNK_METADATA,
            "pass2_method": "sparql-values",
        },
        counts=counts,
        inputs={"documents": documents_path},
        extra={"started_at": started, "finished_at": datetime.now(UTC).isoformat()},
    )
    return counts
