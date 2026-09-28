"""Tests for sentence segmentation. Implements spec 0002 §Decision 4."""

import json

import pytest

from inpnet.corpus.clean import parse_html
from inpnet.manifest import read_jsonl
from inpnet.nlp.segment import run

PARA_A = "Bohr worked in Copenhagen. He met Heisenberg there in 1926."
PARA_B = "Later he moved on. Dr. Pauli joined the institute, e.g. in 1928."
SECTION = f"{PARA_A}\n\n{PARA_B}"


@pytest.fixture
def documents(tmp_path):
    path = tmp_path / "documents.jsonl"
    path.write_text(
        json.dumps(
            {
                "doc_id": "Q7085",
                "qid": "Q7085",
                "title": "Niels Bohr",
                "sections": [{"heading": "", "text": SECTION, "links": []}],
            }
        ),
        encoding="utf-8",
    )
    return path


def test_no_sentence_spans_a_paragraph_break(documents, tmp_path):
    """Regression: 9.1% of sentences used to contain a newline.

    Feeding whole sections to the senter let it run through the blank line, welding
    the last sentence of one paragraph onto the first of the next. That inflates the
    co-occurrence window and invents candidate pairs.
    """
    run(documents, tmp_path / "sentences.jsonl")
    rows = read_jsonl(tmp_path / "sentences.jsonl")
    assert rows, "expected sentences"
    for row in rows:
        assert "\n" not in row["text"], f"sentence spans a paragraph break: {row['text']!r}"


def test_offsets_are_relative_to_section_text(documents, tmp_path):
    """The paragraph base offset must be added back, or link offsets desync."""
    run(documents, tmp_path / "sentences.jsonl")
    for row in read_jsonl(tmp_path / "sentences.jsonl"):
        assert SECTION[row["start"] : row["end"]] == row["text"]


def test_sent_idx_is_continuous_across_paragraphs(documents, tmp_path):
    """sent_idx indexes the section, so it must not restart at each paragraph."""
    run(documents, tmp_path / "sentences.jsonl")
    rows = read_jsonl(tmp_path / "sentences.jsonl")
    assert [r["sent_idx"] for r in rows] == list(range(len(rows)))


def test_abbreviations_do_not_split(documents, tmp_path):
    """'Dr.' and 'e.g.' must not be treated as sentence ends."""
    run(documents, tmp_path / "sentences.jsonl")
    texts = [r["text"] for r in read_jsonl(tmp_path / "sentences.jsonl")]
    assert any("Dr. Pauli joined" in t and "e.g. in 1928" in t for t in texts)


def test_counts_report_paragraphs_and_sections(documents, tmp_path):
    counts = run(documents, tmp_path / "sentences.jsonl")
    assert counts["documents"] == 1
    assert counts["sections"] == 1
    assert counts["paragraphs"] == 2


def test_bracket_split_is_merged_back_together(tmp_path):
    """Regression: senter treats an open "[" as sentence-final more often than
    warranted — a bracketed middle initial inside a name ("Mervyn [M.] Dymally")
    used to split there, losing everything after the "[" to the next "sentence"
    (candidate Q3760460:3:4:Q3760460:Q1922193 in the entity-mention layer, spec
    0002). The two senter fragments must come back as one sentence.
    """
    text = (
        "Carruthers was involved in initiatives such as Project SMART (formed by "
        "Congressman Mervyn [M.] Dymally), the National Society of Black Physicists."
    )
    path = tmp_path / "documents.jsonl"
    path.write_text(
        json.dumps(
            {
                "doc_id": "Q3760460",
                "qid": "Q3760460",
                "title": "T",
                "sections": [{"heading": "", "text": text, "links": []}],
            }
        ),
        encoding="utf-8",
    )
    run(path, tmp_path / "sentences.jsonl")
    rows = read_jsonl(tmp_path / "sentences.jsonl")
    assert len(rows) == 1
    assert rows[0]["text"] == text
    assert not rows[0]["text"].rstrip().endswith("[")


def test_bracket_merge_is_capped(tmp_path):
    """Regression: a genuinely unclosed "[" in the source article (a typo that was
    never fixed — found in a real article) must not swallow the rest of the
    paragraph. The merge gives up after ``MAX_BRACKET_MERGE`` and each remaining
    sentence is still emitted, even though the bracket count stays unbalanced.
    """
    text = " ".join(f"Sentence number {i} here." for i in range(1, 15))
    text = "Born on [September 15, 1931 and grew up nearby. " + text
    path = tmp_path / "documents.jsonl"
    path.write_text(
        json.dumps(
            {
                "doc_id": "Q1",
                "qid": "Q1",
                "title": "T",
                "sections": [{"heading": "", "text": text, "links": []}],
            }
        ),
        encoding="utf-8",
    )
    run(path, tmp_path / "sentences.jsonl")
    rows = read_jsonl(tmp_path / "sentences.jsonl")
    # merging stopped well short of consuming every remaining sentence
    assert len(rows) >= 8


def test_segmentation_composes_with_the_cleaner(tmp_path):
    """End-to-end on real Parsoid shapes: clean -> segment keeps offsets valid.

    The two stages share one coordinate system, so this is the seam most likely to
    drift silently.
    """
    html = (
        '<html><body><section data-mw-section-id="0">'
        '<p>Bohr met <a rel="mw:WikiLink" href="./Werner_Heisenberg">Heisenberg</a>.</p>'
        '<p>Later <a rel="mw:WikiLink" href="./Wolfgang_Pauli">Pauli</a> arrived.</p>'
        "</section></body></html>"
    )
    (section,) = parse_html(html)
    path = tmp_path / "documents.jsonl"
    path.write_text(
        json.dumps(
            {"doc_id": "Q1", "qid": "Q1", "title": "T", "sections": [section]}
        ),
        encoding="utf-8",
    )
    run(path, tmp_path / "sentences.jsonl")

    text = section["text"]
    for row in read_jsonl(tmp_path / "sentences.jsonl"):
        assert text[row["start"] : row["end"]] == row["text"]
        assert "\n" not in row["text"]
    # every link must still fall inside exactly one sentence
    spans = [(r["start"], r["end"]) for r in read_jsonl(tmp_path / "sentences.jsonl")]
    for link in section["links"]:
        assert any(s <= link["start"] < e for s, e in spans), link
