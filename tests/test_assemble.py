"""Tests for schemaless relation record assembly. Implements spec 0005."""

from __future__ import annotations

import json

import pytest

from inpnet.manifest import read_jsonl
from inpnet.relations.assemble import (
    build_relation_record,
    compute_coverage,
    format_relations_summary,
    run,
)
from inpnet.relations.attributes import load_detector_nlp

spacy = pytest.importorskip("spacy")


@pytest.fixture(scope="module")
def nlp():
    return load_detector_nlp()


def _write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


# --- compute_coverage -----------------------------------------------------


def test_compute_coverage_full_when_everything_covered(nlp):
    text = "Fermi worked with Szilard."
    doc = nlp.tokenizer(text)
    # Every content token (Fermi, worked, Szilard) inside some covered span.
    covered = [(0, len(text))]
    assert compute_coverage(doc, covered) == 1.0


def test_compute_coverage_partial(nlp):
    text = "Fermi worked with Szilard."
    doc = nlp.tokenizer(text)
    covered = [(0, 5)]  # just "Fermi"
    coverage = compute_coverage(doc, covered)
    assert 0.0 < coverage < 1.0


def test_compute_coverage_empty_sentence_is_full(nlp):
    doc = nlp.tokenizer("...")  # no content tokens
    assert compute_coverage(doc, []) == 1.0


# --- build_relation_record / uncovered runs -------------------------------


def test_build_relation_record_flags_uncovered_span(nlp):
    text = "He became Deputy Head of Metallurgy under Monty Finniston at Harwell."
    doc = nlp.tokenizer(text)
    candidate = {
        "candidate_id": "Q1:0:0:Qa:Qb",
        "doc_id": "Q1",
        "section_idx": 0,
        "sent_idx": 0,
        "sentence": text,
        "head_entity_id": "Qa",
        "tail_entity_id": "Qb",
    }
    finniston_start = text.index("Monty")
    head = (0, 2, "pronoun")  # "He"
    tail = (finniston_start, finniston_start + len("Monty Finniston"), "span")
    record = build_relation_record(candidate, doc, [], [], head, tail)
    assert record["relation_id"] == "Q1:0:0:Qa:Qb"
    assert record["sentence"] == text
    assert record["coverage"] < 1.0
    assert any("Harwell" in u for u in record["uncovered"])
    assert record["pronoun_resolved"] is True


def test_format_relations_summary_shows_typed_and_unclassified(nlp):
    text = "He became Deputy Head of Metallurgy under Monty Finniston at Harwell."
    doc = nlp.tokenizer(text)
    candidate = {
        "candidate_id": "Q1:0:0:Qa:Qb",
        "doc_id": "Q1",
        "section_idx": 0,
        "sent_idx": 0,
        "sentence": text,
        "head_entity_id": "Qa",
        "tail_entity_id": "Qb",
    }
    record = build_relation_record(candidate, doc, [], [], (0, 2, "name"), None)
    text_out = format_relations_summary([record])
    assert "coverage" in text_out
    assert record["pronoun_resolved"] is False
    assert "unclassified" in text_out


def test_format_relations_summary_empty():
    assert format_relations_summary([]) == "(no relations)"


# --- run() end to end -------------------------------------------------------


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


def _fixture(tmp_path):
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
    attributes_path = tmp_path / "attribute_spans.jsonl"
    _write_jsonl(candidates_path, candidates)
    _write_jsonl(entities_path, [])
    _write_jsonl(
        sentences_path,
        [{"doc_id": "Q1", "section_idx": 0, "sent_idx": 0, "start": 0, "end": len(sentence), "text": sentence}],
    )
    _write_jsonl(
        attributes_path,
        [
            {
                "sentence_id": "Q1:0:0",
                "candidate_id": candidates[0]["candidate_id"],
                "attr_type": "action",
                "value": "worked with",
                "span": [6, 18],
                "detector": "dep_parse",
            }
        ],
    )
    return candidates_path, entities_path, sentences_path, attributes_path


def test_run_writes_relations_and_summaries(tmp_path, nlp):
    candidates_path, entities_path, sentences_path, attributes_path = _fixture(tmp_path)
    out_dir = tmp_path / "out"
    counts = run(
        candidates_path, entities_path, sentences_path, attributes_path, out_dir,
        min_coverage=0.0, nlp=nlp,
    )
    assert counts["n_candidates"] == 1
    relations = read_jsonl(out_dir / "relations.jsonl")
    assert len(relations) == 1
    assert (out_dir / "relations_summary.txt").exists()
    assert (out_dir / "low_coverage_relations_summary.txt").exists()
    assert (out_dir / "_manifest.json").exists()


def test_run_splits_low_coverage(tmp_path, nlp):
    """A very high --min-coverage forces everything into the low-coverage file."""
    candidates_path, entities_path, sentences_path, attributes_path = _fixture(tmp_path)
    out_dir = tmp_path / "out"
    counts = run(
        candidates_path, entities_path, sentences_path, attributes_path, out_dir,
        min_coverage=1.01, nlp=nlp,
    )
    assert counts["n_relations"] == 0
    assert counts["n_low_coverage"] == 1


def test_run_determinism(tmp_path, nlp):
    candidates_path, entities_path, sentences_path, attributes_path = _fixture(tmp_path)
    out_a, out_b = tmp_path / "a", tmp_path / "b"
    run(candidates_path, entities_path, sentences_path, attributes_path, out_a, nlp=nlp)
    run(candidates_path, entities_path, sentences_path, attributes_path, out_b, nlp=nlp)
    assert (out_a / "relations.jsonl").read_bytes() == (out_b / "relations.jsonl").read_bytes()
