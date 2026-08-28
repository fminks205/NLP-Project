"""Tests for the cluster-stage hyperparameter sweep (dev tool, not spec 0003's Interface).

Same fake-embedder strategy as test_cluster.py; importorskip's the same way.
"""

from __future__ import annotations

import json

import pytest

pytest.importorskip("numpy")
pytest.importorskip("umap")
pytest.importorskip("sklearn")

from inpnet.relations.diagnostics import format_exemplars, format_table, sweep  # noqa: E402

np = pytest.importorskip("numpy")


def _fake_embedder(dim: int = 8, seed: int = 0):
    def embed(texts):
        vectors = []
        for i, _ in enumerate(texts):
            rng = np.random.default_rng(seed + i)
            base = 0.0 if i % 2 == 0 else 10.0
            vectors.append(base + rng.normal(scale=0.1, size=dim))
        return np.asarray(vectors)

    return embed


def _candidates_path(tmp_path, n=12):
    rows = [
        {
            "doc_id": "Q1",
            "section_idx": 0,
            "sent_idx": i,
            "sentence": f"sentence number {i}",
            "head_entity_id": "Q1",
            "tail_entity_id": f"Q{i + 2}",
        }
        for i in range(n)
    ]
    path = tmp_path / "candidates.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    return path


def test_sweep_covers_the_full_grid_product(tmp_path):
    candidates = _candidates_path(tmp_path)
    reduction_grid = [
        {"metric": "cosine", "n_neighbors": 3, "n_components": 2},
        {"metric": "cosine", "n_neighbors": 4, "n_components": 3},
    ]
    cluster_grid = [
        {"min_cluster_size": 2, "cluster_selection_method": "eom"},
        {"min_cluster_size": 3, "cluster_selection_method": "leaf"},
    ]

    results = sweep(
        candidates,
        limit=12,
        reduction_grid=reduction_grid,
        cluster_grid=cluster_grid,
        embedder=_fake_embedder(),
    )

    assert len(results) == len(reduction_grid) * len(cluster_grid)
    for row in results:
        assert row["n_sentences"] == 12
        assert 0 <= row["n_noise"] <= 12
        assert 0.0 <= row["largest_cluster_fraction"] <= 1.0
        assert set(row) >= {
            "n_neighbors",
            "n_components",
            "min_cluster_size",
            "cluster_selection_method",
            "n_clusters",
            "n_noise",
            "noise_fraction",
            "largest_cluster_size",
            "largest_cluster_fraction",
            "top_cluster_sizes",
        }


def test_sweep_is_deterministic(tmp_path):
    candidates = _candidates_path(tmp_path)
    grid = [{"min_cluster_size": 2, "cluster_selection_method": "eom"}]
    reduction_grid = [{"metric": "cosine", "n_neighbors": 3, "n_components": 2}]

    a = sweep(
        candidates,
        limit=12,
        reduction_grid=reduction_grid,
        cluster_grid=grid,
        embedder=_fake_embedder(),
    )
    b = sweep(
        candidates,
        limit=12,
        reduction_grid=reduction_grid,
        cluster_grid=grid,
        embedder=_fake_embedder(),
    )
    assert a == b


def test_format_table_orders_by_largest_cluster_fraction_ascending():
    results = [
        {
            "n_neighbors": 15,
            "n_components": 5,
            "min_cluster_size": 10,
            "cluster_selection_method": "eom",
            "n_clusters": 1,
            "n_noise": 0,
            "noise_fraction": 0.0,
            "largest_cluster_size": 100,
            "largest_cluster_fraction": 0.99,
            "top_cluster_sizes": [100],
        },
        {
            "n_neighbors": 15,
            "n_components": 5,
            "min_cluster_size": 5,
            "cluster_selection_method": "leaf",
            "n_clusters": 6,
            "n_noise": 4,
            "noise_fraction": 0.04,
            "largest_cluster_size": 20,
            "largest_cluster_fraction": 0.2,
            "top_cluster_sizes": [20, 18, 17],
        },
    ]
    table = format_table(results)
    lines = table.splitlines()
    # header + separator, then the balanced (0.2) row before the degenerate (0.99) one
    balanced_line_no = next(i for i, l in enumerate(lines) if "0.2" in l)
    degenerate_line_no = next(i for i, l in enumerate(lines) if "0.99" in l)
    assert balanced_line_no < degenerate_line_no


def test_format_table_handles_empty_results():
    """Regression: an empty grid used to raise KeyError building column widths."""
    assert format_table([]) == "(no results)"


def test_sweep_expand_top_n_attaches_exemplars_only_to_the_winners(tmp_path):
    candidates = _candidates_path(tmp_path)
    reduction_grid = [
        {"metric": "cosine", "n_neighbors": 3, "n_components": 2},
        {"metric": "cosine", "n_neighbors": 4, "n_components": 3},
    ]
    cluster_grid = [
        {"min_cluster_size": 2, "cluster_selection_method": "eom"},
        {"min_cluster_size": 3, "cluster_selection_method": "leaf"},
    ]

    results = sweep(
        candidates,
        limit=12,
        reduction_grid=reduction_grid,
        cluster_grid=cluster_grid,
        embedder=_fake_embedder(),
        expand_top_n=2,
    )

    expanded = [r for r in results if "exemplars_by_cluster" in r]
    assert len(expanded) == 2
    # every other combo has no exemplars_by_cluster key at all
    assert len(results) - len(expanded) == 2

    # expansion picked the two most-balanced (lowest largest_cluster_fraction) combos
    by_fraction = sorted(results, key=lambda r: r["largest_cluster_fraction"])
    expected_ids = {id(r) for r in by_fraction[:2]}
    assert {id(r) for r in expanded} == expected_ids

    for r in expanded:
        assert r["exemplars_by_cluster"]
        for cluster in r["exemplars_by_cluster"]:
            assert cluster["exemplar_sentences"]
            assert all(isinstance(s, str) for s in cluster["exemplar_sentences"])


def test_sweep_without_expand_top_n_attaches_no_exemplars(tmp_path):
    candidates = _candidates_path(tmp_path)
    results = sweep(
        candidates,
        limit=12,
        reduction_grid=[{"metric": "cosine", "n_neighbors": 3, "n_components": 2}],
        cluster_grid=[{"min_cluster_size": 2, "cluster_selection_method": "eom"}],
        embedder=_fake_embedder(),
    )
    assert all("exemplars_by_cluster" not in r for r in results)


def test_format_exemplars_reports_when_nothing_expanded():
    assert format_exemplars([{"largest_cluster_fraction": 0.5}]).startswith("(no combos")


def test_format_exemplars_renders_expanded_combo():
    results = [
        {
            "n_neighbors": 15,
            "n_components": 5,
            "metric": "cosine",
            "min_cluster_size": 5,
            "cluster_selection_method": "leaf",
            "n_clusters": 1,
            "largest_cluster_fraction": 0.2,
            "exemplars_by_cluster": [
                {"cluster_id": 0, "size": 3, "exemplar_sentences": ["a", "b"]},
            ],
        }
    ]
    text = format_exemplars(results)
    assert "cluster 0 (size 3)" in text
    assert "- a" in text
    assert "- b" in text
