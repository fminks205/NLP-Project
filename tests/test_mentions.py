"""Tests for the mention layer and candidate pairs. Implements spec 0002 §Decision 3, 5."""

import json

import pytest

from inpnet.manifest import read_jsonl
from inpnet.nlp.mentions import (
    RULE_LINK_LINK,
    RULE_SUBJECT_LINK,
    IncompleteCache,
    check_cache_coverage,
    load_human_index,
    locate_sentence,
    run,
)


def _index(rows):
    rows = sorted(rows, key=lambda r: r["start"])
    return {("Q1", 0): {"starts": [r["start"] for r in rows], "rows": rows}}


def _sent(sent_idx, start, end, text="s"):
    return {
        "doc_id": "Q1", "section_idx": 0, "sent_idx": sent_idx,
        "start": start, "end": end, "text": text,
    }


def test_locate_sentence_finds_containing_sentence():
    index = _index([_sent(0, 0, 20), _sent(1, 22, 40)])
    assert locate_sentence(index, ("Q1", 0), 25)["sent_idx"] == 1
    assert locate_sentence(index, ("Q1", 0), 0)["sent_idx"] == 0


def test_locate_sentence_returns_none_in_gaps():
    """Offsets between sentences (the '\\n\\n' join) belong to no sentence."""
    index = _index([_sent(0, 0, 20), _sent(1, 22, 40)])
    assert locate_sentence(index, ("Q1", 0), 21) is None


def test_locate_sentence_returns_none_before_first_and_unknown_section():
    index = _index([_sent(0, 5, 20)])
    assert locate_sentence(index, ("Q1", 0), 1) is None
    assert locate_sentence(index, ("Q9", 3), 10) is None


def test_load_human_index_keeps_only_humans(tmp_path):
    path = tmp_path / "cache.jsonl"
    path.write_text(
        "\n".join(
            json.dumps(r)
            for r in [
                {"title": "Niels Bohr", "qid": "Q7085", "is_human": True, "resolved_title": "Niels Bohr"},
                {"title": "Physics", "qid": "Q413", "is_human": False, "resolved_title": "Physics"},
                {"title": "Nowhere", "qid": None, "is_human": False, "resolved_title": None},
            ]
        ),
        encoding="utf-8",
    )
    index = load_human_index(path)
    assert set(index) == {"Niels Bohr"}


# -- end-to-end over a tiny fixture corpus -------------------------------

SENTENCE = "Bohr worked with Werner Heisenberg and Wolfgang Pauli."


@pytest.fixture
def corpus(tmp_path):
    docs = tmp_path / "documents.jsonl"
    docs.write_text(
        json.dumps(
            {
                "doc_id": "Q7085", "qid": "Q7085", "title": "Niels Bohr",
                "revision_id": 1, "url": "u", "fetched_at": "t", "license": "l",
                "sections": [
                    {
                        "heading": "", "text": SENTENCE,
                        "links": [
                            {"start": 17, "end": 34, "surface": "Werner Heisenberg",
                             "target_title": "Werner Heisenberg"},
                            {"start": 39, "end": 53, "surface": "Wolfgang Pauli",
                             "target_title": "Wolfgang Pauli"},
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    sents = tmp_path / "sentences.jsonl"
    sents.write_text(
        json.dumps({"doc_id": "Q7085", "section_idx": 0, "sent_idx": 0,
                    "start": 0, "end": len(SENTENCE), "text": SENTENCE}),
        encoding="utf-8",
    )
    cache = tmp_path / "cache.jsonl"
    cache.write_text(
        "\n".join(
            json.dumps(r) for r in [
                {"title": "Werner Heisenberg", "resolved_title": "Werner Heisenberg",
                 "qid": "Q40276", "is_human": True, "birth_year": 1901,
                 "death_year": 1976, "gender": "Q6581097", "occupations": ["Q169470"]},
                {"title": "Wolfgang Pauli", "resolved_title": "Wolfgang Pauli",
                 "qid": "Q65989", "is_human": True, "birth_year": 1900,
                 "death_year": 1958, "gender": "Q6581097", "occupations": ["Q169470"]},
            ]
        ),
        encoding="utf-8",
    )
    seed = tmp_path / "seed.jsonl"
    seed.write_text(
        json.dumps({"qid": "Q7085", "title": "Niels Bohr", "birth_year": 1885,
                    "death_year": 1962, "gender": "Q6581097"}),
        encoding="utf-8",
    )
    return docs, sents, cache, seed, tmp_path


def test_end_to_end_mentions_and_pairs(corpus):
    docs, sents, cache, seed, out = corpus
    counts = run(docs, sents, cache, seed, out)

    assert counts["link_mentions"] == 2
    assert counts["subject_mentions"] == 1
    # subject pairs with each linked person; the two linked people pair with each other
    assert counts[RULE_SUBJECT_LINK] == 2
    assert counts[RULE_LINK_LINK] == 1
    assert counts["docs_without_pairs"] == 0


def test_mention_offsets_round_trip(corpus):
    docs, sents, cache, seed, out = corpus
    run(docs, sents, cache, seed, out)
    for mention in read_jsonl(out / "mentions.jsonl"):
        if mention["mention_type"] != "link":
            continue
        actual = SENTENCE[mention["start"] : mention["end"]]
        assert actual == mention["surface"]
        assert actual == actual.strip(), "span must not carry surrounding whitespace"


def test_entities_flag_seed_membership(corpus):
    docs, sents, cache, seed, out = corpus
    run(docs, sents, cache, seed, out)
    entities = {e["qid"]: e for e in read_jsonl(out / "entities.jsonl")}
    assert entities["Q7085"]["in_seed"] is True
    assert entities["Q40276"]["in_seed"] is False
    assert entities["Q7085"]["birth_year"] == 1885, "seed metadata should be preferred"
    assert entities["Q40276"]["birth_year"] == 1901, "non-seed falls back to the resolver"


def test_candidates_record_their_rule(corpus):
    docs, sents, cache, seed, out = corpus
    run(docs, sents, cache, seed, out)
    cands = read_jsonl(out / "candidates.jsonl")
    rules = {c["rule"] for c in cands}
    assert rules == {RULE_SUBJECT_LINK, RULE_LINK_LINK}
    for cand in cands:
        assert cand["sentence"] == SENTENCE
        if cand["rule"] == RULE_SUBJECT_LINK:
            assert cand["head_span"] is None, "the subject has no span until coref exists"


def test_run_is_deterministic(corpus):
    docs, sents, cache, seed, out = corpus
    run(docs, sents, cache, seed, out)
    first = (out / "candidates.jsonl").read_bytes()
    run(docs, sents, cache, seed, out)
    assert (out / "candidates.jsonl").read_bytes() == first


def test_incomplete_cache_is_rejected(corpus, tmp_path):
    """A short cache must fail loudly, not emit a near-empty graph.

    This is the exact hazard that existed on disk: a 200-row cache left over from a
    `resolve --limit 200` test run would otherwise have produced a plausible-looking
    but almost empty network.
    """
    docs, sents, _, seed, out = corpus
    empty = tmp_path / "empty_cache.jsonl"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(IncompleteCache, match="absent"):
        run(docs, sents, empty, seed, out)


def test_lenient_allows_incomplete_cache(corpus, tmp_path):
    docs, sents, _, seed, out = corpus
    empty = tmp_path / "empty_cache.jsonl"
    empty.write_text("", encoding="utf-8")
    counts = run(docs, sents, empty, seed, out, strict=False)
    assert counts["non_person_links"] == 2
    assert counts["link_mentions"] == 0
    assert counts["docs_without_pairs"] == 1


def test_coverage_guard_rejects_unresolved_claims(corpus, tmp_path):
    """A title present but never looked up must not pass as 'not a person'."""
    docs, sents, _, seed, out = corpus
    cache = tmp_path / "partial.jsonl"
    cache.write_text(
        "\n".join(
            json.dumps(r)
            for r in [
                {"title": "Werner Heisenberg", "qid": "Q1", "is_human": True,
                 "claims_resolved": True, "resolved_title": "Werner Heisenberg"},
                {"title": "Wolfgang Pauli", "qid": "Q2", "is_human": None,
                 "claims_resolved": False, "resolved_title": "Wolfgang Pauli"},
            ]
        ),
        encoding="utf-8",
    )
    report = check_cache_coverage(docs, cache, strict=False)
    assert report["claims_unresolved"] == 1
    with pytest.raises(IncompleteCache, match="unresolved"):
        check_cache_coverage(docs, cache, strict=True)


def test_coverage_guard_passes_on_complete_cache(corpus):
    docs, _, cache, _, _ = corpus
    report = check_cache_coverage(docs, cache, strict=True)
    assert report["absent_from_cache"] == 0
    assert report["claims_unresolved"] == 0
