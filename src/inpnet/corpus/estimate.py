"""Estimate stage: project download size before fetching anything.

Answers "how big is this going to be?" without doing the fetch. Two parts:

* **Exact**, not estimated: the Action API's ``prop=info`` returns each page's
  wikitext ``length`` in bytes, 50 titles per request. For a 15k-article corpus
  that is ~304 requests, so the total wikitext size is *measured*.
* **Sampled**: Parsoid HTML is larger than wikitext by a ratio that has to be
  observed. A sample of articles is fetched and the ratio applied to the exact
  wikitext total.

The sample must be drawn from the real seed list. Measured 2026-08-12: famous
physicists (Einstein, Curie) average ~120 KB of wikitext, while a random sample
of the seed set averages ~10 KB. Estimating from famous articles overshoots by
roughly 10x.
"""

from __future__ import annotations

import random
import statistics
from pathlib import Path
from typing import Any

from ..manifest import read_jsonl, write_manifest
from ..wiki import WikiClient


def human_bytes(n: float) -> str:
    """Format a byte count for humans."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024:
            return f"{n:,.1f} {unit}"
        n /= 1024
    return f"{n:,.1f} PB"


def run(
    seed_path: Path,
    *,
    contact: str,
    sample_size: int = 30,
    limit: int | None = None,
    delay: float = 1.0,
    seed: int = 0,
    out_dir: Path | None = None,
) -> dict[str, Any]:
    """Measure wikitext size for the whole seed list and project HTML size."""
    people = read_jsonl(seed_path)
    if limit:
        people = people[:limit]
    titles = [p["title"] for p in people]

    client = WikiClient(contact, delay=delay)

    # --- exact wikitext totals ------------------------------------------
    lengths: list[int] = []
    missing: list[str] = []
    redirected: list[str] = []
    for info in client.page_info(titles):
        if info["missing"]:
            missing.append(info["title"])
            continue
        if info["redirected_from"]:
            redirected.append(info["redirected_from"])
        if info["length"] is not None:
            lengths.append(info["length"])

    total_wikitext = sum(lengths)

    # --- sampled HTML:wikitext ratio ------------------------------------
    rng = random.Random(seed)
    sample_titles = rng.sample(titles, min(sample_size, len(titles)))
    ratios: list[float] = []
    sampled_html = 0
    by_title = {i["title"]: i for i in client.page_info(sample_titles)}
    for title in sample_titles:
        info = by_title.get(title)
        if not info or info["missing"] or not info["length"]:
            continue
        page = client.page_html(title)
        sampled_html += len(page.html)
        ratios.append(len(page.html) / info["length"])

    ratio_mean = statistics.fmean(ratios) if ratios else float("nan")
    ratio_median = statistics.median(ratios) if ratios else float("nan")

    # Weighted ratio is the honest one for projecting a total: it weights each
    # article by its size instead of letting tiny articles dominate the mean.
    sampled_wikitext = sum(
        by_title[t]["length"] for t in sample_titles if by_title.get(t, {}).get("length")
    )
    ratio_weighted = sampled_html / sampled_wikitext if sampled_wikitext else float("nan")

    projected_html = total_wikitext * ratio_weighted

    result = {
        "articles": len(lengths),
        "missing": len(missing),
        "redirected": len(redirected),
        "total_wikitext_bytes": total_wikitext,
        "mean_wikitext_bytes": int(statistics.fmean(lengths)) if lengths else 0,
        "median_wikitext_bytes": int(statistics.median(lengths)) if lengths else 0,
        "max_wikitext_bytes": max(lengths) if lengths else 0,
        "sample_size": len(ratios),
        "html_ratio_weighted": round(ratio_weighted, 3),
        "html_ratio_mean": round(ratio_mean, 3),
        "html_ratio_median": round(ratio_median, 3),
        "projected_html_bytes": int(projected_html),
        "projected_html_human": human_bytes(projected_html),
        "missing_titles": missing[:20],
    }

    if out_dir:
        write_manifest(
            out_dir,
            stage="estimate",
            config={"seed_path": str(seed_path), "sample_size": sample_size, "seed": seed},
            counts=result,
            inputs={"seed": seed_path},
        )
    return result
