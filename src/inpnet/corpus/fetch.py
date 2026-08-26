"""Fetch stage: seed.jsonl -> data/raw/html/{qid}.html. Implements spec 0001 §Decision 2.

Raw HTML is written unmodified so cleaning can be re-run and iterated without
re-fetching a single article — the whole reason fetch and clean are separate
commands.

The run is **resumable**. At ~15k articles and one polite request per second a
full fetch is an overnight job, and a job that cannot survive an interruption is
a job that will be run twice. Already-downloaded files are skipped, and every
attempt is appended to fetch_log.jsonl so failures are data rather than lost.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..manifest import read_jsonl, write_manifest
from ..wiki import PERMALINK, WikiClient, WikiError


def run(
    seed_path: Path,
    out_dir: Path,
    *,
    contact: str,
    limit: int | None = None,
    delay: float = 1.0,
    resume: bool = True,
    progress_every: int = 25,
) -> dict[str, Any]:
    """Fetch article HTML for the seed list. Returns summary counts."""
    people = read_jsonl(seed_path)
    if limit:
        people = people[:limit]

    html_dir = out_dir / "html"
    html_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "fetch_log.jsonl"

    client = WikiClient(contact, delay=delay)
    counts = {"ok": 0, "skipped": 0, "error": 0}

    with log_path.open("a", encoding="utf-8", newline="\n") as log:
        for index, person in enumerate(people, start=1):
            qid, title = person["qid"], person["title"]
            target = html_dir / f"{qid}.html"

            if resume and target.exists() and target.stat().st_size > 0:
                counts["skipped"] += 1
                continue

            entry: dict[str, Any] = {
                "qid": qid,
                "title": title,
                "fetched_at": datetime.now(UTC).isoformat(),
            }
            try:
                page = client.page_html(title)
            except WikiError as exc:
                counts["error"] += 1
                entry |= {"status": "error", "revision_id": None, "error": str(exc)}
            else:
                target.write_bytes(page.html)
                counts["ok"] += 1
                entry |= {
                    "status": "ok",
                    "revision_id": page.revision_id,
                    "url": PERMALINK.format(revid=page.revision_id),
                    "bytes": len(page.html),
                }
            log.write(json.dumps(entry, ensure_ascii=False) + "\n")
            log.flush()

            if progress_every and index % progress_every == 0:
                done = counts["ok"] + counts["skipped"]
                print(f"  {index}/{len(people)}  ok={counts['ok']} "
                      f"skipped={counts['skipped']} error={counts['error']}", flush=True)

    write_manifest(
        out_dir,
        stage="fetch",
        spec="0001-corpus-acquisition",
        config={
            "seed_path": str(seed_path),
            "limit": limit,
            "delay": delay,
            "resume": resume,
        },
        counts=counts | {"requested": len(people)},
        inputs={"seed": seed_path},
    )
    return counts
