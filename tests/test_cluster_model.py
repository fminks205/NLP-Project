"""Exercises the real sentence-transformers model. Implements spec 0003 §Decision 2.

Separated from test_cluster.py and marked `model` (deselect with `-m 'not model'`) since
it needs the `relations` extra installed *and* a one-time ~80MB download of
`all-MiniLM-L6-v2` on first run -- the same reason live-Wikimedia tests carry the
existing `network` marker instead of running by default.
"""

from __future__ import annotations

import pytest

pytest.importorskip("sentence_transformers")

from inpnet.relations.cluster import DEFAULT_MODEL, _default_embedder  # noqa: E402


@pytest.mark.model
def test_default_embedder_produces_one_vector_per_sentence():
    embed = _default_embedder(DEFAULT_MODEL)
    texts = [
        "Bohr worked with Heisenberg in Copenhagen.",
        "Curie was married to Pierre Curie.",
    ]
    vectors = embed(texts)
    assert len(vectors) == len(texts)
    assert len(vectors[0]) == len(vectors[1]) > 0


@pytest.mark.model
def test_default_embedder_places_similar_sentences_closer():
    """Sanity check that this is actually a semantic embedding, not noise."""
    import numpy as np

    embed = _default_embedder(DEFAULT_MODEL)
    texts = [
        "He was her doctoral advisor.",
        "She supervised his PhD thesis.",
        "They collaborated on a paper about photosynthesis.",
    ]
    vectors = np.asarray(embed(texts))

    def cosine(a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    advisor_pair = cosine(vectors[0], vectors[1])
    unrelated_pair = cosine(vectors[0], vectors[2])
    assert advisor_pair > unrelated_pair
