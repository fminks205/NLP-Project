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
from inpnet.relations.cluster import (
    PERSON_PLACEHOLDER,
    apply_masking,
    build_entity_index,
    build_sentence_starts,
    dedup_sentences,
    format_cluster_summary,
    mask_sentence,
    run,
    sentence_key,
)

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


# --- entity masking (spec 0003 §Decision 2a) ----------------------------------


def test_mask_sentence_masks_linked_spans():
    text = "He worked with Niels Bohr on the theory."
    masked = mask_sentence(text, spans=[(15, 25)], names=[])
    assert masked == f"He worked with {PERSON_PLACEHOLDER} on the theory."


def test_mask_sentence_masks_full_name_and_surname():
    text = "Henry B. Eyring was sustained as an apostle; Eyring later served as commissioner."
    masked = mask_sentence(text, spans=[], names=["Henry B. Eyring", "Eyring"])
    assert "Eyring" not in masked
    assert masked.count(PERSON_PLACEHOLDER) == 2


def test_mask_sentence_does_not_match_inside_another_word():
    """Word-boundary matching: a surname that's a prefix of an unrelated word must not fire."""
    text = "He lived near Eyringville."
    masked = mask_sentence(text, spans=[], names=["Eyring"])
    assert masked == text


def test_mask_sentence_spans_and_names_combine():
    text = "Eyring was sustained after the death of Howard W. Hunter."
    start = text.index("Howard")
    end = start + len("Howard W. Hunter")
    masked = mask_sentence(text, spans=[(start, end)], names=["Eyring"])
    assert masked == (
        f"{PERSON_PLACEHOLDER} was sustained after the death of {PERSON_PLACEHOLDER}."
    )


def test_mask_sentence_drops_out_of_range_spans_instead_of_corrupting_text():
    """Regression test: a span outside `text`'s own bounds -- e.g. a section-relative
    offset fed in by mistake instead of a sentence-local one, spec 0002 §Decision 4's
    actual convention -- must not silently mutate unrelated text. Python slicing doesn't
    raise on an out-of-range index, it clips, which is exactly how this slipped through
    once: the placeholder got appended at the very end instead of removing anything, and
    the name was left fully intact. See spec 0003 Changelog.
    """
    text = "Short sentence."
    masked = mask_sentence(text, spans=[(358, 374)], names=[])
    assert masked == text


def _mention(doc_id, section_idx, sent_idx, start, end, surface, entity_id, mention_type="link"):
    return {
        "mention_id": f"{doc_id}:{section_idx}:{start if start is not None else 'subject'}",
        "doc_id": doc_id,
        "section_idx": section_idx,
        "sent_idx": sent_idx,
        "start": start,
        "end": end,
        "surface": surface,
        "entity_id": entity_id,
        "mention_type": mention_type,
        "link_method": "wikilink" if mention_type == "link" else "subject",
        "confidence": 1.0,
    }


def _entity(qid, canonical_name):
    return {
        "qid": qid,
        "canonical_name": canonical_name,
        "in_seed": True,
        "mention_count": 1,
        "doc_count": 1,
        "doc_ids": [],
        "birth_year": None,
        "death_year": None,
        "gender": None,
        "occupations": [],
    }


def _sentence(doc_id, section_idx, sent_idx, start, end, text):
    """A `sentences.jsonl` row (spec 0002) -- `start`/`end` are this sentence's own
    section-relative offsets, the thing `build_sentence_starts` reads."""
    return {
        "doc_id": doc_id,
        "section_idx": section_idx,
        "sent_idx": sent_idx,
        "start": start,
        "end": end,
        "text": text,
    }


def test_build_entity_index_converts_section_relative_spans_to_sentence_local():
    """Regression test, mirroring the real bug (see spec 0003 Changelog): a link
    mention's `start`/`end` are section-relative (spec 0002 §Decision 4), not
    sentence-relative. "Sir Humphry Davy" mentioned as the *other* party (not the
    article's own subject) in a sentence that isn't the first in its section -- exactly
    the shape of the real failure, where the section-relative offset, fed straight into
    slicing against the short sentence string, silently no-opped instead of masking
    anything.
    """
    doc_id = "Q320889"
    section_start = 291  # nonzero: this sentence is not the first in its section
    text = (
        "He was awarded the Davy Medal, named for the great British chemist "
        "Sir Humphry Davy, by the Royal Society of London in 1907."
    )
    local_start = text.index("Sir Humphry Davy")
    local_end = local_start + len("Sir Humphry Davy")

    mention = _mention(
        doc_id,
        3,
        3,
        section_start + local_start,
        section_start + local_end,
        "Sir Humphry Davy",
        "Q131761",
    )
    sentence_starts = build_sentence_starts(
        [_sentence(doc_id, 3, 3, section_start, section_start + len(text), text)]
    )

    entity_index = build_entity_index([mention], [], sentence_starts)
    masked = apply_masking(
        [{"sentence_id": f"{doc_id}:3:3", "doc_id": doc_id, "sentence": text}], entity_index
    )[0]

    assert masked["masked_sentence"] == text.replace("Sir Humphry Davy", PERSON_PLACEHOLDER)


def test_build_entity_index_and_apply_masking_reproduces_eyring_example():
    """The three real sentences that motivated spec 0003 §Decision 2a: all name only the
    article subject (no other linked person), so unmasked they collapse to "Eyring ...
    Eyring ... Eyring" plus generic biographical phrasing -- entity bias, not relation
    similarity. Masked, the subject disappears from all three and each sentence's own
    distinguishing content is what's left.
    """
    doc_id = "Q949425"
    sentences = [
        {
            "sentence_id": f"{doc_id}:0:0",
            "doc_id": doc_id,
            "section_idx": 0,
            "sent_idx": 0,
            "sentence": (
                "Following the death of church president Howard W. Hunter, Eyring was "
                "sustained as a member of the church's Quorum of the Twelve Apostles on "
                "April 1, 1995 and ordained an apostle later that week."
            ),
        },
        {
            "sentence_id": f"{doc_id}:0:1",
            "doc_id": doc_id,
            "section_idx": 0,
            "sent_idx": 1,
            "sentence": (
                "Eyring served as president of Ricks College from 1971 to 1977, as a "
                "counselor to Presiding Bishop Robert D. Hales from 1985 to 1992, and as "
                "a member of the First Quorum of the Seventy, from 1992 to 1995."
            ),
        },
        {
            "sentence_id": f"{doc_id}:0:2",
            "doc_id": doc_id,
            "section_idx": 0,
            "sent_idx": 2,
            "sentence": (
                "Eyring has served twice as commissioner of church education, from "
                "September 1980 to April 1985, and from September 1992 to January 2005, "
                "when he was replaced by W. Rolfe Kerr."
            ),
        },
    ]
    mentions = [_mention(doc_id, None, None, None, None, "Henry B. Eyring", "Q949425", "subject")]
    entities = [_entity("Q949425", "Henry B. Eyring")]

    # no link mentions in this example, so no section-relative spans to convert
    entity_index = build_entity_index(mentions, entities, {})
    masked = apply_masking(sentences, entity_index)

    for row in masked:
        assert "Eyring" not in row["masked_sentence"]
        assert PERSON_PLACEHOLDER in row["masked_sentence"]
        assert "Eyring" in row["sentence"]  # original text untouched

    # no longer near-duplicates of each other on the subject's name alone
    texts = [row["masked_sentence"] for row in masked]
    assert len(set(texts)) == 3


def test_apply_masking_is_noop_without_entity_index():
    sentences = [{"sentence_id": "Q1:0:0", "doc_id": "Q1", "sentence": "Eyring did X."}]
    masked = apply_masking(sentences, None)
    assert masked[0]["masked_sentence"] == masked[0]["sentence"]


def test_run_embeds_masked_text_using_section_relative_spans(tmp_path):
    """Same doc/subject as `_sample_candidates`, plus one sentence naming both the
    subject and a linked person -- filler sentences keep UMAP/HDBSCAN well-behaved (see
    the other `run()` tests' n=12), and don't mention "Eyring" so masking leaves them
    alone. The mention span is deliberately encoded with a nonzero section offset baked
    in (not equal to its sentence-local offset), so this test can't pass by accident the
    way it would if `run()` still sliced section-relative offsets directly against
    sentence-local text (see spec 0003 Changelog).
    """
    doc_id = "Q1"
    target = "Eyring worked with Niels Bohr."
    local_start = target.index("Niels Bohr")
    local_end = local_start + len("Niels Bohr")
    section_start = 500  # nonzero -- not the first sentence in its section
    rows = [_candidate(doc_id, 0, i, f"sentence number {i}", "Q1", f"Q{i + 2}") for i in range(11)]
    rows.append(_candidate(doc_id, 0, 11, target, "Q1", "Q2"))
    candidates_path = _candidates_path(tmp_path, rows)

    mentions_path = tmp_path / "mentions.jsonl"
    with mentions_path.open("w", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                _mention(
                    doc_id,
                    0,
                    11,
                    section_start + local_start,
                    section_start + local_end,
                    "Niels Bohr",
                    "Q2",
                )
            )
            + "\n"
        )
        handle.write(
            json.dumps(_mention(doc_id, None, None, None, None, "Eyring", "Q1", "subject"))
            + "\n"
        )

    entities_path = tmp_path / "entities.jsonl"
    with entities_path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(_entity("Q1", "Eyring")) + "\n")
        handle.write(json.dumps(_entity("Q2", "Niels Bohr")) + "\n")

    sentences_path = tmp_path / "sentences.jsonl"
    with sentences_path.open("w", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                _sentence(doc_id, 0, 11, section_start, section_start + len(target), target)
            )
            + "\n"
        )

    seen_texts = {}

    def spy_embedder(texts):
        seen_texts["texts"] = list(texts)
        return _fake_embedder()(texts)

    run(
        candidates_path,
        tmp_path / "out",
        mentions_path=mentions_path,
        entities_path=entities_path,
        sentences_path=sentences_path,
        min_cluster_size=3,
        n_neighbors=3,
        n_components=2,
        embedder=spy_embedder,
    )

    assert seen_texts["texts"][-1] == f"{PERSON_PLACEHOLDER} worked with {PERSON_PLACEHOLDER}."
    assert seen_texts["texts"][:-1] == [f"sentence number {i}" for i in range(11)]

    manifest = json.loads((tmp_path / "out" / "_manifest.json").read_text(encoding="utf-8"))
    assert manifest["config"]["mask_entities"] is True
    assert manifest["counts"]["n_sentences_masked"] == 1
    assert "mentions" in manifest["inputs"]
    assert "entities" in manifest["inputs"]
    assert "sentences" in manifest["inputs"]


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


# --- format_cluster_summary --------------------------------------------------


def _summary_row(cluster_id, size, *, exemplars=None, samples=None, verdict=None, label=None):
    return {
        "cluster_id": cluster_id,
        "clustering_run_id": "run1",
        "size": size,
        "exemplar_sentences": exemplars or [f"exemplar for {cluster_id}"],
        "sample_sentences": samples or [],
        "coherence_reviewed_by": [],
        "coherence_verdict": verdict,
        "human_label": label,
    }


def test_format_cluster_summary_empty():
    assert format_cluster_summary([]) == "(no clusters)"


def test_format_cluster_summary_orders_by_cluster_id():
    rows = [_summary_row(370, 12), _summary_row(6, 40)]
    text = format_cluster_summary(rows)
    assert text.index("cluster   6") < text.index("cluster 370")


def test_format_cluster_summary_aligns_columns_across_blocks():
    """The `size` field starts at the same character column in every cluster's header,
    even though cluster_id and size widths individually vary."""
    rows = [_summary_row(6, 7), _summary_row(1168, 62)]
    lines = format_cluster_summary(rows).splitlines()
    headers = [line for line in lines if line.startswith("cluster")]
    assert len(headers) == 2
    assert headers[0].index("size") == headers[1].index("size")


def test_format_cluster_summary_indents_sentences_one_level_with_aligned_bullets():
    rows = [
        _summary_row(1, 2, exemplars=["short one"]),
        _summary_row(22, 33, exemplars=["another sentence"]),
    ]
    lines = format_cluster_summary(rows).splitlines()
    bullet_lines = [line for line in lines if "- " in line and not line.startswith("cluster")]
    assert bullet_lines  # at least one exemplar line per cluster
    # every sentence line starts its bullet at the same column, regardless of that
    # cluster's id/size width
    bullet_columns = {line.index("-") for line in bullet_lines}
    assert bullet_columns == {4}


def test_format_cluster_summary_samples_are_opt_in():
    rows = [_summary_row(1, 2, exemplars=["ex"], samples=["sample sentence"])]
    without = format_cluster_summary(rows)
    with_samples = format_cluster_summary(rows, show_samples=True)
    assert "sample sentence" not in without
    assert "sample sentence" in with_samples


def test_format_cluster_summary_shows_label_when_present():
    rows = [_summary_row(1, 2, label="mentorship-ish")]
    assert "mentorship-ish" in format_cluster_summary(rows)
