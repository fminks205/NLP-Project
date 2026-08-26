"""Tests for the relation-clustering interface. Implements spec 0003 §Decision 1-5.

Dedup/join logic is plain Python and runs unconditionally. Everything past embedding
needs the optional `relations` extra (numpy, umap-learn, scikit-learn); those tests
`importorskip` rather than hard-failing when it isn't installed. None of them touch the
real sentence-transformers model or the network -- see test_cluster_model.py for that,
gated behind the `model` marker instead.
"""

from __future__ import annotations

import json

import pytest

from inpnet.manifest import read_jsonl
from inpnet.relations.cluster import dedup_sentences, run, sentence_key

np = pytest.importorskip("numpy")
pytest.importorskip("umap")
pytest.importorskip("sklearn")


def _candidate(doc_id, section_idx, sent_idx, sentence, head, tail):
    return {
        "candidate_id": f"{doc_id}:{section_idx}:{sent_idx}:{head}:{tail}",
        "doc_id": doc_id,
        "section_idx": section_idx,
        "sent_idx": sent_idx,
        "sentence": sentence,
        "head_entity_id": head,
        "tail_entity_id": tail,
        "rule": "subject_link",
        "people_in_sentence": 2,
    }


def _candidates_path(tmp_path, rows):
    path = tmp_path / "candidates.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    return path


def _fake_embedder(dim: int = 8, seed: int = 0):
    """Deterministic stand-in for sentence-transformers: two well-separated blobs.

    Index-based, not content-based -- good enough to exercise dedup -> embed -> reduce
    -> cluster end to end without a model, matching spec 0003's chosen test strategy.
    Every call with the same `texts` returns the same array, which is what the
    determinism test relies on.
    """

    def embed(texts):
        vectors = []
        for i, _ in enumerate(texts):
            rng = np.random.default_rng(seed + i)
            base = 0.0 if i % 2 == 0 else 10.0
            vectors.append(base + rng.normal(scale=0.1, size=dim))
        return np.asarray(vectors)

    return embed


# --- dedup / join -----------------------------------------------------------


def test_dedup_sentences_collapses_shared_sentence():
    """A three-person sentence yields three candidate pairs but one sentence."""
    rows = [
        _candidate("Q1", 0, 0, "Bohr, Heisenberg and Pauli met in Copenhagen.", "Q1", "Q2"),
        _candidate("Q1", 0, 0, "Bohr, Heisenberg and Pauli met in Copenhagen.", "Q1", "Q3"),
        _candidate("Q1", 0, 0, "Bohr, Heisenberg and Pauli met in Copenhagen.", "Q2", "Q3"),
        _candidate("Q1", 1, 0, "He later moved to Cambridge.", "Q1", "Q4"),
    ]
    sentences = dedup_sentences(rows)
    assert len(sentences) == 2
    assert {s["sentence_id"] for s in sentences} == {"Q1:0:0", "Q1:1:0"}


def test_dedup_sentences_first_seen_order():
    rows = [
        _candidate("Q2", 0, 1, "second", "Q1", "Q2"),
        _candidate("Q1", 0, 0, "first", "Q1", "Q2"),
        _candidate("Q2", 0, 1, "second", "Q1", "Q3"),  # duplicate key, later
    ]
    sentences = dedup_sentences(rows)
    assert [s["sentence_id"] for s in sentences] == ["Q2:0:1", "Q1:0:0"]


def test_sentence_key_matches_candidate_and_relation_cluster_rows():
    """The join key `run()` writes to `relation_clusters.jsonl` must match a candidate's key."""
    candidate = _candidate("Q1", 2, 3, "text", "Q1", "Q2")
    assert sentence_key(candidate) == "Q1:2:3"


# --- full stage, fake embedder -----------------------------------------------


def _sample_candidates(tmp_path, n=12):
    rows = [
        _candidate("Q1", 0, i, f"sentence number {i}", "Q1", f"Q{i + 2}") for i in range(n)
    ]
    return _candidates_path(tmp_path, rows)


def test_run_end_to_end_with_fake_embedder(tmp_path):
    candidates = _sample_candidates(tmp_path)
    out_dir = tmp_path / "out"

    counts = run(
        candidates,
        out_dir,
        min_cluster_size=3,
        n_neighbors=3,
        n_components=2,
        embedder=_fake_embedder(),
    )

    assert counts["n_sentences"] == 12
    assert counts["n_clusters"] >= 1
    assert 0 <= counts["n_noise"] <= 12

    sentence_ids = read_jsonl(out_dir / "sentence_ids.jsonl")
    clusters = read_jsonl(out_dir / "relation_clusters.jsonl")
    summary = read_jsonl(out_dir / "cluster_summary.jsonl")
    manifest = json.loads((out_dir / "_manifest.json").read_text(encoding="utf-8"))

    assert len(sentence_ids) == len(clusters) == 12
    assert {c["sentence_id"] for c in clusters} == {s["sentence_id"] for s in sentence_ids}
    assert (out_dir / "sentence_embeddings.npy").exists()
    embeddings = np.load(out_dir / "sentence_embeddings.npy")
    assert embeddings.shape[0] == 12

    # every non-noise cluster_id in relation_clusters.jsonl has a summary row
    summary_ids = {row["cluster_id"] for row in summary}
    for row in clusters:
        if row["cluster_id"] != -1:
            assert row["cluster_id"] in summary_ids

    assert manifest["spec"] == "0003-relation-typology"
    assert manifest["stage"] == "cluster"
    assert manifest["counts"]["clustering_run_id"] == counts["clustering_run_id"]


def test_run_is_deterministic(tmp_path):
    """Rerun on unchanged input reproduces identical cluster assignments (AGENTS.md §5)."""
    candidates = _sample_candidates(tmp_path)

    counts_a = run(
        candidates,
        tmp_path / "out_a",
        min_cluster_size=3,
        n_neighbors=3,
        n_components=2,
        embedder=_fake_embedder(),
    )
    counts_b = run(
        candidates,
        tmp_path / "out_b",
        min_cluster_size=3,
        n_neighbors=3,
        n_components=2,
        embedder=_fake_embedder(),
    )

    assert counts_a["clustering_run_id"] == counts_b["clustering_run_id"]
    clusters_a = read_jsonl(tmp_path / "out_a" / "relation_clusters.jsonl")
    clusters_b = read_jsonl(tmp_path / "out_b" / "relation_clusters.jsonl")
    assert clusters_a == clusters_b


def test_clustering_run_id_changes_with_hyperparameters(tmp_path):
    """`cluster_id` is run-scoped (spec 0003 §Decision 4): a different config, different run."""
    candidates = _sample_candidates(tmp_path)

    counts_a = run(
        candidates,
        tmp_path / "out_a",
        min_cluster_size=3,
        n_neighbors=3,
        n_components=2,
        embedder=_fake_embedder(),
    )
    counts_b = run(
        candidates,
        tmp_path / "out_b",
        min_cluster_size=4,  # only this changed
        n_neighbors=3,
        n_components=2,
        embedder=_fake_embedder(),
    )

    assert counts_a["clustering_run_id"] != counts_b["clustering_run_id"]


def test_run_respects_limit(tmp_path):
    candidates = _sample_candidates(tmp_path, n=12)
    counts = run(
        candidates,
        tmp_path / "out",
        min_cluster_size=2,
        n_neighbors=2,
        n_components=2,
        limit=5,
        embedder=_fake_embedder(),
    )
    assert counts["n_sentences"] == 5
