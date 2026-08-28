"""Field-agnostic sentence clustering. Implements spec 0003 §Decision 1-5.

Groups the sentences behind `candidates.jsonl` by embedding, then UMAP, then HDBSCAN --
no relation labels presupposed. A candidate pair inherits its sentence's `cluster_id` by
joining on `sentence_id`; `candidates.jsonl` itself is not rewritten (spec 0002's stage
boundary: no stage overwrites another stage's output).

Heavy dependencies (sentence-transformers, umap-learn, scikit-learn) are imported lazily
inside the functions that need them, not at module level. That keeps this module
importable -- and the dependency-free parts of the test suite runnable -- without the
optional `relations` extra installed, and lets an injected `embedder` (spec 0003's own
test strategy) stand in for the real model.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from ..manifest import read_jsonl, write_jsonl, write_manifest

SPEC = "0003-relation-typology"

# Proposals to validate against the real corpus, not measured defaults -- see spec 0003
# §Decision 2-3 and §Open questions. All are overridable from the CLI.
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_MIN_CLUSTER_SIZE = 15
DEFAULT_N_NEIGHBORS = 15
DEFAULT_N_COMPONENTS = 5
DEFAULT_RANDOM_STATE = 42
DEFAULT_METRIC = "cosine"
DEFAULT_CLUSTER_SELECTION_METHOD = "eom"
DEFAULT_N_EXEMPLARS = 5
DEFAULT_N_SAMPLES = 5

# (texts) -> array-like of shape (len(texts), dim). The real implementation wraps
# sentence-transformers; tests inject a small deterministic stand-in instead, per the
# "fake embedder" test strategy so the fast suite never needs a model download.
Embedder = Callable[[Sequence[str]], Any]


def sentence_key(row: dict) -> str:
    """The `(doc_id, section_idx, sent_idx)` key a candidate pair's sentence lives at."""
    return f"{row['doc_id']}:{row['section_idx']}:{row['sent_idx']}"


def prepare_sentences(candidates: list[dict], *, limit: int | None = None) -> list[dict]:
    """Dedup already-loaded candidates to distinct sentences, sorted and (optionally) capped.

    Shared by `run()` and the diagnostics sweep (`diagnostics.py`) so both operate on
    the exact same sentence set for a given `limit` -- a sweep result is only a useful
    predictor of the full run if it's built from the same loading logic. Takes an
    already-loaded `candidates` list, not a path, so a caller that also needs the raw
    candidate count (as `run()` does, for its manifest) reads the file once.
    """
    sentences = dedup_sentences(candidates)
    # Sort by the (doc_id, section_idx, sent_idx) triple, not the concatenated
    # sentence_id string -- string order puts "…:10" before "…:2" once an index reaches
    # double digits, which is still deterministic but needlessly hard to read.
    sentences.sort(key=lambda row: (row["doc_id"], row["section_idx"], row["sent_idx"]))
    if limit is not None:
        sentences = sentences[:limit]
    return sentences


def dedup_sentences(candidates: list[dict]) -> list[dict]:
    """One row per distinct sentence behind the candidate pairs, first-seen order.

    Spec 0003 §Decision 1: candidate pairs collapse to far fewer distinct sentences
    (measured on the full corpus: 43,664 of 76,205) because a sentence naming three or
    more people produces several pairs. Clustering runs once per sentence; every
    candidate pair rejoins its sentence's cluster by `sentence_id`.
    """
    seen: dict[str, dict] = {}
    for row in candidates:
        key = sentence_key(row)
        if key not in seen:
            seen[key] = {
                "sentence_id": key,
                "doc_id": row["doc_id"],
                "section_idx": row["section_idx"],
                "sent_idx": row["sent_idx"],
                "sentence": row["sentence"],
            }
    return list(seen.values())


def _default_embedder(model_name: str) -> Embedder:
    """Build a real sentence-transformers embedder. Needs the `relations` extra."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:  # pragma: no cover - exercised by the model-marked test
        raise ImportError(
            "sentence-transformers is not installed. Run `uv sync --extra relations`."
        ) from exc

    model = SentenceTransformer(model_name)

    def embed(texts: Sequence[str]) -> Any:
        return model.encode(list(texts), show_progress_bar=False, convert_to_numpy=True)

    return embed


def reduce_dimensions(
    embeddings: Any,
    *,
    n_neighbors: int = DEFAULT_N_NEIGHBORS,
    n_components: int = DEFAULT_N_COMPONENTS,
    random_state: int = DEFAULT_RANDOM_STATE,
    metric: str = DEFAULT_METRIC,
) -> Any:
    """UMAP dimensionality reduction ahead of density-based clustering (spec 0003 §Decision 3).

    `n_neighbors`/`n_components` are clamped to the input size so this also runs on the
    small samples the test suite uses, not just the full corpus.

    `metric="cosine"` (not UMAP's Euclidean default) because sentence-transformer
    embeddings are trained and conventionally compared by cosine similarity, not
    Euclidean distance -- a mismatch discovered the hard way on the first full-corpus
    run, which produced one 43,634-sentence cluster instead of anything relation-like.
    """
    try:
        import umap
    except ImportError as exc:  # pragma: no cover - exercised by extras-gated tests
        raise ImportError("umap-learn is not installed. Run `uv sync --extra relations`.") from exc

    n = len(embeddings)
    reducer = umap.UMAP(
        n_neighbors=min(n_neighbors, max(n - 1, 2)),
        n_components=min(n_components, max(n - 1, 1)),
        random_state=random_state,
        metric=metric,
    )
    return reducer.fit_transform(embeddings)


def cluster_embeddings(
    reduced: Any,
    *,
    min_cluster_size: int = DEFAULT_MIN_CLUSTER_SIZE,
    cluster_selection_method: str = DEFAULT_CLUSTER_SELECTION_METHOD,
) -> tuple[Any, Any]:
    """HDBSCAN over the reduced embeddings. Label `-1` is noise (spec 0003 §Decision 3).

    Uses `sklearn.cluster.HDBSCAN` (scikit-learn >= 1.3) rather than the standalone
    `hdbscan` package -- same algorithm, but it ships prebuilt wheels alongside
    scikit-learn instead of needing its own compiled build, which matters on a fresh
    Python 3.13 install. See the spec's Alternatives table for the reasoning; the choice
    of package wasn't pinned by the spec itself.

    `cluster_selection_method="eom"` (excess of mass, HDBSCAN's default) picks the most
    *persistent* clusters in the condensed tree, which on the first full-corpus run
    picked the root -- one 43k-sentence cluster -- over any smaller substructure.
    `"leaf"` selects the leaves of that tree instead: more, smaller clusters, no
    persistence competition against one dominant blob. Exposed as a parameter rather
    than hardcoded because which is right is an empirical question the diagnostics
    sweep (`diagnostics.py`) exists to answer, not a settled default.
    """
    try:
        from sklearn.cluster import HDBSCAN
    except ImportError as exc:  # pragma: no cover - exercised by extras-gated tests
        raise ImportError(
            "scikit-learn>=1.3 is not installed. Run `uv sync --extra relations`."
        ) from exc

    # copy=True: `reduced` is read again afterward (build_cluster_summary's centroid
    # distances), so it must not be mutated in place. Explicit because sklearn's default
    # is changing (False -> True in 1.10) and the old default silently risked exactly that.
    clusterer = HDBSCAN(
        min_cluster_size=min(min_cluster_size, max(len(reduced), 2)),
        cluster_selection_method=cluster_selection_method,
        copy=True,
    )
    labels = clusterer.fit_predict(reduced)
    probabilities = getattr(clusterer, "probabilities_", None)
    if probabilities is None:  # pragma: no cover - defensive; sklearn always sets this
        probabilities = [1.0] * len(labels)
    return labels, probabilities


def build_cluster_summary(
    sentences: list[dict],
    labels: Any,
    reduced: Any,
    *,
    clustering_run_id: str,
    n_exemplars: int = DEFAULT_N_EXEMPLARS,
    n_samples: int = DEFAULT_N_SAMPLES,
    seed: int = DEFAULT_RANDOM_STATE,
) -> list[dict]:
    """One row per non-noise cluster: size, exemplars nearest centroid, a random sample.

    Exemplars and the random sample are what the human-review step (spec 0003
    §Decision 5) actually reads to judge cluster coherence; `coherence_verdict` and
    `human_label` start `null` and are filled in by that review, not by this function.
    """
    import numpy as np

    by_cluster: dict[int, list[int]] = {}
    for idx, label in enumerate(labels):
        if int(label) == -1:
            continue
        by_cluster.setdefault(int(label), []).append(idx)

    rng = random.Random(seed)
    summary = []
    for cluster_id in sorted(by_cluster):
        indices = by_cluster[cluster_id]
        points = np.asarray([reduced[i] for i in indices])
        centroid = points.mean(axis=0)
        nearest_first = sorted(
            indices, key=lambda i: float(np.linalg.norm(np.asarray(reduced[i]) - centroid))
        )
        exemplars = [sentences[i]["sentence"] for i in nearest_first[:n_exemplars]]

        sample_pool = list(indices)
        rng.shuffle(sample_pool)
        samples = [sentences[i]["sentence"] for i in sample_pool[:n_samples]]

        summary.append(
            {
                "cluster_id": cluster_id,
                "clustering_run_id": clustering_run_id,
                "size": len(indices),
                "exemplar_sentences": exemplars,
                "sample_sentences": samples,
                "coherence_reviewed_by": [],
                "coherence_verdict": None,
                "human_label": None,
            }
        )
    return summary


def format_cluster_summary(rows: list[dict], *, show_samples: bool = False) -> str:
    """Render `cluster_summary.jsonl` rows for a human to read during the review step
    (spec 0003 §Decision 5) -- presentation only, `cluster_summary.jsonl` itself is
    untouched and stays the JSONL other tooling reads.

    Layout: one block per cluster, ordered by `cluster_id`. Each block's header
    (`cluster N   size N   verdict N`) right-aligns its numbers to the widest value in
    the whole file, so the same field lines up in the same column from block to block --
    scan down the page and the `size` column reads as a column, not a ragged edge.
    Sentences sit one indentation level under their cluster's header, each behind the
    same `- ` bullet, so every sentence's text starts at the same column too, in every
    cluster, not just within one block. `human_label` is free text with no natural
    bound, so it prints on its own line rather than forcing every header to pad out to
    the longest label in the file.
    """
    if not rows:
        return "(no clusters)"

    ordered = sorted(rows, key=lambda r: r["cluster_id"])
    id_width = max(len(str(r["cluster_id"])) for r in ordered)
    size_width = max(len(str(r["size"])) for r in ordered)
    verdict_width = max(len(str(r.get("coherence_verdict") or "-")) for r in ordered)

    blocks = []
    for r in ordered:
        cluster_id = str(r["cluster_id"]).rjust(id_width)
        size = str(r["size"]).rjust(size_width)
        verdict = str(r.get("coherence_verdict") or "-").ljust(verdict_width)
        lines = [f"cluster {cluster_id}   size {size}   verdict {verdict}"]
        if r.get("human_label"):
            lines.append(f"    label: {r['human_label']}")

        lines.append("    exemplars:")
        for sentence in r["exemplar_sentences"]:
            lines.append(f"    - {sentence}")
        if show_samples and r.get("sample_sentences"):
            lines.append("    samples:")
            for sentence in r["sample_sentences"]:
                lines.append(f"    - {sentence}")

        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def _clustering_run_id(config: dict[str, Any]) -> str:
    """Deterministic short id for a run, derived from its config.

    Spec 0003 §Decision 4: `cluster_id` is run-scoped, and a typology plug-in records
    the `clustering_run_id` it targets. Deriving the id from config rather than a
    timestamp means an unchanged config always yields the same id, so a rerun with the
    same settings produces a byte-identical manifest -- the determinism requirement in
    AGENTS.md §5 -- instead of a merely equivalent one.
    """
    blob = json.dumps(config, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:12]


def run(
    candidates_path: Path,
    out_dir: Path,
    *,
    model_name: str = DEFAULT_MODEL,
    min_cluster_size: int = DEFAULT_MIN_CLUSTER_SIZE,
    n_neighbors: int = DEFAULT_N_NEIGHBORS,
    n_components: int = DEFAULT_N_COMPONENTS,
    random_state: int = DEFAULT_RANDOM_STATE,
    metric: str = DEFAULT_METRIC,
    cluster_selection_method: str = DEFAULT_CLUSTER_SELECTION_METHOD,
    limit: int | None = None,
    embedder: Embedder | None = None,
) -> dict[str, Any]:
    """Run the `cluster` stage end to end. Implements spec 0003 §Interface.

    `embedder`, when given, replaces the real sentence-transformers model -- used by
    tests to exercise the full dedup -> embed -> reduce -> cluster -> summarize pipeline
    without a model download. Everything downstream of embedding (UMAP, HDBSCAN) still
    needs the `relations` extra installed either way.
    """
    import numpy as np

    candidates = read_jsonl(candidates_path)
    sentences = prepare_sentences(candidates, limit=limit)

    config = {
        "model_name": model_name,
        "min_cluster_size": min_cluster_size,
        "n_neighbors": n_neighbors,
        "n_components": n_components,
        "random_state": random_state,
        "metric": metric,
        "cluster_selection_method": cluster_selection_method,
        "limit": limit,
        "n_sentences": len(sentences),
    }
    clustering_run_id = _clustering_run_id(config)

    embed = embedder or _default_embedder(model_name)
    texts = [row["sentence"] for row in sentences]
    embeddings = np.asarray(embed(texts))

    reduced = reduce_dimensions(
        embeddings,
        n_neighbors=n_neighbors,
        n_components=n_components,
        random_state=random_state,
        metric=metric,
    )
    labels, probabilities = cluster_embeddings(
        reduced,
        min_cluster_size=min_cluster_size,
        cluster_selection_method=cluster_selection_method,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "sentence_embeddings.npy", embeddings)
    write_jsonl(
        out_dir / "sentence_ids.jsonl",
        [{"sentence_id": row["sentence_id"]} for row in sentences],
    )
    write_jsonl(
        out_dir / "relation_clusters.jsonl",
        [
            {
                "sentence_id": row["sentence_id"],
                "cluster_id": int(label),
                "cluster_prob": float(prob),
            }
            for row, label, prob in zip(sentences, labels, probabilities, strict=True)
        ],
    )
    summary = build_cluster_summary(
        sentences,
        labels,
        reduced,
        clustering_run_id=clustering_run_id,
        seed=random_state,
    )
    write_jsonl(out_dir / "cluster_summary.jsonl", summary)

    n_noise = int(sum(1 for label in labels if int(label) == -1))
    counts = {
        "n_candidates": len(candidates),
        "n_sentences": len(sentences),
        "n_clusters": len(summary),
        "n_noise": n_noise,
        "noise_fraction": round(n_noise / len(sentences), 4) if sentences else 0.0,
        "clustering_run_id": clustering_run_id,
    }
    write_manifest(
        out_dir,
        stage="cluster",
        spec=SPEC,
        config=config,
        counts=counts,
        inputs={"candidates": candidates_path},
        extra={"clustering_run_id": clustering_run_id},
    )
    return counts
