"""Fast hyperparameter diagnostics for the `cluster` stage. Not part of spec 0003's
Interface -- a dev tool for answering a question the spec explicitly left open ("does
UMAP + HDBSCAN produce coherent clusters on this corpus?") without paying a full
43,664-sentence run (~3.5 minutes) per hyperparameter attempt.

Embeds a sample once, then sweeps a grid of settings against that one embedding. UMAP's
fit is the expensive part; HDBSCAN on an already-reduced embedding is cheap. So the grid
is nested: outer loop varies UMAP's own parameters (`n_neighbors`, `n_components`,
`metric` -- one `reduce_dimensions` call each), inner loop varies HDBSCAN's
(`min_cluster_size`, `cluster_selection_method` -- reused against the same reduction).

Stays strictly inside the UMAP + HDBSCAN family spec 0003 §Decision 3 already commits
to -- it tunes the hyperparameters the spec itself calls "proposals to validate," not a
different algorithm. Produces a recommendation for which flags to pass `inpnet cluster`,
not a pipeline artifact: nothing here is written under `data/interim/`.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from ..manifest import read_jsonl
from .cluster import (
    DEFAULT_MODEL,
    DEFAULT_N_EXEMPLARS,
    DEFAULT_RANDOM_STATE,
    Embedder,
    _default_embedder,
    apply_masking,
    build_cluster_summary,
    build_entity_index,
    build_sentence_starts,
    cluster_embeddings,
    prepare_sentences,
    reduce_dimensions,
)

# Six UMAP reductions (the expensive step) x eight HDBSCAN settings each (cheap) = 48
# rows, but only six actual UMAP fits. `metric` stays fixed at "cosine" -- the first
# full-corpus run already showed it's not the dominant lever (see cluster.py).
DEFAULT_REDUCTION_GRID: list[dict[str, Any]] = [
    {"metric": "cosine", "n_neighbors": n_neighbors, "n_components": n_components}
    for n_neighbors in (15, 30)
    for n_components in (5, 15, 30)
]
DEFAULT_CLUSTER_GRID: list[dict[str, Any]] = [
    {"min_cluster_size": min_cluster_size, "cluster_selection_method": method}
    for min_cluster_size in (5, 10, 20, 40)
    for method in ("eom", "leaf")
]


def _summarize(
    reduction_params: dict[str, Any], cluster_params: dict[str, Any], labels: Any
) -> dict[str, Any]:
    n = len(labels)
    counts = Counter(int(label) for label in labels)
    n_noise = counts.pop(-1, 0)
    cluster_sizes = sorted(counts.values(), reverse=True)
    largest = cluster_sizes[0] if cluster_sizes else 0
    return {
        **reduction_params,
        **cluster_params,
        "n_sentences": n,
        "n_clusters": len(cluster_sizes),
        "n_noise": n_noise,
        "noise_fraction": round(n_noise / n, 4) if n else 0.0,
        "largest_cluster_size": largest,
        "largest_cluster_fraction": round(largest / n, 4) if n else 0.0,
        "top_cluster_sizes": cluster_sizes[:5],
    }


def sweep(
    candidates_path: Path,
    *,
    limit: int = 5000,
    model_name: str = DEFAULT_MODEL,
    random_state: int = DEFAULT_RANDOM_STATE,
    mentions_path: Path | None = None,
    entities_path: Path | None = None,
    sentences_path: Path | None = None,
    reduction_grid: list[dict[str, Any]] | None = None,
    cluster_grid: list[dict[str, Any]] | None = None,
    embedder: Embedder | None = None,
    expand_top_n: int = 0,
    n_exemplars: int = DEFAULT_N_EXEMPLARS,
    max_clusters_to_expand: int = 5,
) -> list[dict[str, Any]]:
    """Embed a sample once, try a grid of UMAP/HDBSCAN settings, report structure per combo.

    `limit` samples from the same `prepare_sentences` a full `run()` would use (same
    sort order, so `--limit 5000` here is the *first* 5,000 sentences a real run with
    `--limit 5000` would see too) -- a sweep result is only informative if it's reading
    the same data a real run would. `mentions_path`/`entities_path`/`sentences_path`
    apply the same entity masking `run()` does (spec 0003 §Decision 2a) for the same
    reason -- pass all three, or leave them None to sweep on unmasked text.

    By default this only ever computes size statistics -- no sentence text -- so a
    result gives you a number to rank combos by but nothing to actually read. Pass
    `expand_top_n > 0` to additionally attach `exemplars_by_cluster` (real sentences,
    nearest-to-centroid, from `build_cluster_summary` -- the same logic `run()` uses) to
    the `expand_top_n` most-balanced combos (lowest `largest_cluster_fraction`), capped
    at `max_clusters_to_expand` of that combo's largest clusters. Every other result row
    has no `exemplars_by_cluster` key at all. This reuses each combo's already-computed
    UMAP reduction and HDBSCAN labels rather than rerunning either -- only
    `build_cluster_summary`'s cheap centroid-distance sort is extra work.
    """
    import numpy as np

    candidates = read_jsonl(candidates_path)
    sentences = prepare_sentences(candidates, limit=limit)

    entity_index = None
    if mentions_path is not None and entities_path is not None and sentences_path is not None:
        sentence_starts = build_sentence_starts(read_jsonl(sentences_path))
        entity_index = build_entity_index(
            read_jsonl(mentions_path), read_jsonl(entities_path), sentence_starts
        )
    sentences = apply_masking(sentences, entity_index)

    embed = embedder or _default_embedder(model_name)
    embeddings = np.asarray(embed([row["masked_sentence"] for row in sentences]))

    reduction_grid = reduction_grid if reduction_grid is not None else DEFAULT_REDUCTION_GRID
    cluster_grid = cluster_grid if cluster_grid is not None else DEFAULT_CLUSTER_GRID

    results: list[dict[str, Any]] = []
    reduced_and_labels: list[tuple[Any, Any]] = []  # parallel to results, for expansion
    for reduction_params in reduction_grid:
        reduced = reduce_dimensions(embeddings, random_state=random_state, **reduction_params)
        for cluster_params in cluster_grid:
            labels, _probabilities = cluster_embeddings(reduced, **cluster_params)
            results.append(_summarize(reduction_params, cluster_params, labels))
            reduced_and_labels.append((reduced, labels))

    if expand_top_n:
        ranking = sorted(range(len(results)), key=lambda i: results[i]["largest_cluster_fraction"])
        for i in ranking[:expand_top_n]:
            reduced, labels = reduced_and_labels[i]
            summary = build_cluster_summary(
                sentences,
                labels,
                reduced,
                clustering_run_id="sweep",
                n_exemplars=n_exemplars,
                n_samples=0,
                seed=random_state,
            )
            biggest_first = sorted(summary, key=lambda row: row["size"], reverse=True)
            results[i]["exemplars_by_cluster"] = [
                {
                    "cluster_id": row["cluster_id"],
                    "size": row["size"],
                    "exemplar_sentences": row["exemplar_sentences"],
                }
                for row in biggest_first[:max_clusters_to_expand]
            ]

    return results


def format_table(results: list[dict[str, Any]]) -> str:
    """Render sweep results as a table, most-balanced (lowest largest-cluster share) first.

    "Most balanced" is a heuristic for "worth a human look," not a correctness measure
    -- it flags the one-giant-cluster failure mode from the first full-corpus run, but a
    low largest-cluster-fraction with plausible-looking cluster count is still a
    candidate to eyeball with exemplar sentences before trusting it, not a final answer.
    """
    if not results:
        return "(no results)"

    ranked = sorted(results, key=lambda r: r["largest_cluster_fraction"])
    columns = [
        "n_neighbors",
        "n_components",
        "min_cluster_size",
        "cluster_selection_method",
        "n_clusters",
        "n_noise",
        "noise_fraction",
        "largest_cluster_fraction",
        "top_cluster_sizes",
    ]
    widths = {c: max(len(c), *(len(str(r[c])) for r in ranked)) for c in columns}
    header = "  ".join(c.ljust(widths[c]) for c in columns)
    lines = [header, "-" * len(header)]
    for r in ranked:
        lines.append("  ".join(str(r[c]).ljust(widths[c]) for c in columns))
    return "\n".join(lines)


def format_exemplars(results: list[dict[str, Any]]) -> str:
    """Render the `exemplars_by_cluster` attached by `sweep(..., expand_top_n=...)`.

    Only rows carrying that key print anything; call this after `format_table` so a
    reader sees the ranking first and the actual sentences for the top rows second.
    """
    expanded = [r for r in results if "exemplars_by_cluster" in r]
    if not expanded:
        return "(no combos expanded -- call sweep(..., expand_top_n=N) first)"

    ranked = sorted(expanded, key=lambda r: r["largest_cluster_fraction"])
    blocks = []
    for r in ranked:
        settings = ", ".join(
            f"{k}={r[k]}"
            for k in (
                "n_neighbors",
                "n_components",
                "metric",
                "min_cluster_size",
                "cluster_selection_method",
            )
        )
        header = (
            f"[{settings}] -- {r['n_clusters']} clusters, "
            f"largest {r['largest_cluster_fraction']:.1%}"
        )
        lines = [header]
        for cluster in r["exemplars_by_cluster"]:
            lines.append(f"  cluster {cluster['cluster_id']} (size {cluster['size']}):")
            for sentence in cluster["exemplar_sentences"]:
                lines.append(f"    - {sentence}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)
