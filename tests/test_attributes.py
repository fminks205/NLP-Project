"""Tests for relation attribute span detection. Implements spec 0004."""

from __future__ import annotations

import json

import pytest

from inpnet.manifest import read_jsonl
from inpnet.relations.attributes import (
    _pronoun_span,
    detect_ner_attributes,
    load_detector_nlp,
    locate_participant_span,
    run,
    sentence_key,
    shortest_dependency_span,
)

spacy = pytest.importorskip("spacy")


@pytest.fixture(scope="module")
def nlp():
    return load_detector_nlp()


def _write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


# --- sentence_key -------------------------------------------------------


def test_sentence_key_matches_candidate_row():
    row = {"doc_id": "Q1", "section_idx": 2, "sent_idx": 5}
    assert sentence_key(row) == "Q1:2:5"


# --- locate_participant_span --------------------------------------------


def test_locate_participant_span_from_section_relative_offset():
    """A linked participant's span is section-relative and must be converted by
    subtracting the sentence's own start -- spec 0003 §2a's conversion, reused."""
    text = "He worked with Fermi at Chicago."
    # In the full section, this sentence starts at offset 100, and "Fermi" sits at
    # section-relative [115, 120) -- i.e. sentence-local [15, 20).
    start, end, method = locate_participant_span(
        text, "Q123", [115, 120], sentence_start=100, canonical_names={}
    )
    assert text[start:end] == "Fermi"
    assert method == "span"


def test_locate_participant_span_out_of_bounds_returns_none():
    text = "Short sentence."
    assert (
        locate_participant_span(text, "Q123", [900, 905], sentence_start=100, canonical_names={})
        is None
    )


def test_locate_participant_span_subject_by_name():
    """The article subject has no span (spec 0002 §Decision 3) -- located by name,
    reusing spec 0003 §2a's `_name_variants` matching."""
    text = "Eyring served as president of Ricks College."
    start, end, method = locate_participant_span(
        text, "Q1", None, sentence_start=0, canonical_names={"Q1": "Henry B. Eyring"}
    )
    assert text[start:end] == "Eyring"
    assert method == "name"


def test_locate_participant_span_subject_name_not_found_falls_back_to_nothing():
    """No name match and no pronoun in the sentence at all -- genuinely unresolvable."""
    text = "A sentence naming nobody by name or pronoun."
    assert (
        locate_participant_span(
            text, "Q1", None, sentence_start=0, canonical_names={"Q1": "Henry B. Eyring"}
        )
        is None
    )


def test_locate_participant_span_falls_back_to_pronoun():
    """Diagnostic finding: 42% of all candidate pairs are `subject_link` pairs whose
    subject is referred to only by pronoun in-sentence. Accepted-for-now heuristic:
    assume the pronoun is the subject -- flagged via `method == "pronoun"`."""
    text = "He did doctoral research under Owen Richardson."
    result = locate_participant_span(
        text, "Q1", None, sentence_start=0, canonical_names={"Q1": "Henry B. Eyring"}
    )
    assert result is not None
    start, end, method = result
    assert text[start:end].lower() == "he"
    assert method == "pronoun"


def test_pronoun_span_male_gender():
    text = "Later she introduced him to the committee."
    span = _pronoun_span(text, "Q6581097")  # male
    assert text[span[0] : span[1]].lower() == "him"


def test_pronoun_span_female_gender():
    text = "Later he introduced her to the committee."
    span = _pronoun_span(text, "Q6581072")  # female
    assert text[span[0] : span[1]].lower() == "her"


def test_pronoun_span_unknown_gender_takes_earliest_of_either_set():
    text = "Later she introduced him to the committee."
    span = _pronoun_span(text, None)
    assert text[span[0] : span[1]].lower() == "she"


def test_pronoun_span_none_when_no_pronoun_present():
    assert _pronoun_span("A sentence with no third-person pronoun at all.", None) is None


# --- shortest_dependency_span -------------------------------------------


def test_shortest_dependency_span_finds_connecting_verb_phrase(nlp):
    text = "Fermi worked with Szilard on the reactor."
    doc = nlp(text)
    head = (0, len("Fermi"))
    tail_start = text.index("Szilard")
    tail = (tail_start, tail_start + len("Szilard"))
    span = shortest_dependency_span(doc, head, tail)
    assert span is not None
    start, end = span
    value = text[start:end]
    assert "worked" in value


def test_shortest_dependency_span_same_token_returns_none(nlp):
    doc = nlp("Fermi met Fermi.")  # degenerate on purpose
    span = shortest_dependency_span(doc, (0, 5), (0, 5))
    assert span is None


# --- detect_ner_attributes ------------------------------------------------


def test_detect_ner_attributes_maps_labels_to_attr_types(nlp):
    doc = nlp("In 1955 he moved to Harwell to join AERE.")
    attrs = detect_ner_attributes(doc)
    types = {a["attr_type"] for a in attrs}
    # At least one date should surface as `time`; NER label mapping is deterministic,
    # exact place/org recall on this toy sentence is not asserted (spec 0004 Risks).
    assert "time" in types
    for a in attrs:
        assert a["value"] == doc.text[a["span"][0] : a["span"][1]]
        assert a["detector"] == "spacy_ner"


# --- run() end to end -----------------------------------------------------


def _candidate(doc_id, section_idx, sent_idx, sentence, head, tail, head_span, tail_span):
    return {
        "candidate_id": f"{doc_id}:{section_idx}:{sent_idx}:{head}:{tail}",
        "doc_id": doc_id,
        "section_idx": section_idx,
        "sent_idx": sent_idx,
        "sentence": sentence,
        "head_entity_id": head,
        "tail_entity_id": tail,
        "head_span": head_span,
        "tail_span": tail_span,
        "rule": "link_link",
        "people_in_sentence": 2,
    }


def test_run_writes_spans_and_manifest(tmp_path, nlp):
    sentence = "Fermi worked with Szilard on the reactor."
    szilard_start = sentence.index("Szilard")
    candidates = [
        _candidate(
            "Q1", 0, 0, sentence, "Q_fermi", "Q_szilard",
            head_span=[0, 5], tail_span=[szilard_start, szilard_start + 7],
        )
    ]
    candidates_path = tmp_path / "candidates.jsonl"
    entities_path = tmp_path / "entities.jsonl"
    sentences_path = tmp_path / "sentences.jsonl"
    _write_jsonl(candidates_path, candidates)
    _write_jsonl(entities_path, [])
    _write_jsonl(
        sentences_path,
        [{"doc_id": "Q1", "section_idx": 0, "sent_idx": 0, "start": 0, "end": len(sentence), "text": sentence}],
    )

    out_dir = tmp_path / "out"
    counts = run(candidates_path, entities_path, sentences_path, out_dir, nlp=nlp)

    spans = read_jsonl(out_dir / "attribute_spans.jsonl")
    assert counts["n_candidates"] == 1
    assert counts["n_sentences"] == 1
    action_spans = [s for s in spans if s["attr_type"] == "action"]
    assert len(action_spans) == 1
    assert action_spans[0]["candidate_id"] == candidates[0]["candidate_id"]
    assert (out_dir / "_manifest.json").exists()


def test_run_determinism(tmp_path, nlp):
    """Two runs on unchanged input produce byte-identical output (AGENTS.md §5)."""
    sentence = "Fermi worked with Szilard on the reactor."
    szilard_start = sentence.index("Szilard")
    candidates = [
        _candidate(
            "Q1", 0, 0, sentence, "Q_fermi", "Q_szilard",
            head_span=[0, 5], tail_span=[szilard_start, szilard_start + 7],
        )
    ]
    candidates_path = tmp_path / "candidates.jsonl"
    entities_path = tmp_path / "entities.jsonl"
    sentences_path = tmp_path / "sentences.jsonl"
    _write_jsonl(candidates_path, candidates)
    _write_jsonl(entities_path, [])
    _write_jsonl(
        sentences_path,
        [{"doc_id": "Q1", "section_idx": 0, "sent_idx": 0, "start": 0, "end": len(sentence), "text": sentence}],
    )

    out_a, out_b = tmp_path / "a", tmp_path / "b"
    run(candidates_path, entities_path, sentences_path, out_a, nlp=nlp)
    run(candidates_path, entities_path, sentences_path, out_b, nlp=nlp)
    a = (out_a / "attribute_spans.jsonl").read_bytes()
    b = (out_b / "attribute_spans.jsonl").read_bytes()
    assert a == b
