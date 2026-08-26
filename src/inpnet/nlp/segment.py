"""Sentence segmentation. Implements spec 0002 §Decision 4.

spaCy `en_core_web_sm` with only the `senter` pipe: a 12 MB model, CPU-only, and it
handles the abbreviations this corpus is full of ("Dr.", "et al.", "Ph.D.") that a regex
splitter gets wrong.

Offsets are **relative to section text**, so the wiki-link offsets from spec 0001 remain
valid against the same coordinate system.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

from ..manifest import write_manifest

MODEL = "en_core_web_sm"


def load_senter(model: str = MODEL):
    """Load a sentence-splitting-only spaCy pipeline.

    Everything except sentence boundaries is excluded — the tagger, parser, NER and
    lemmatizer all cost time we have no use for here.
    """
    import spacy

    nlp = spacy.load(
        model, exclude=["parser", "ner", "lemmatizer", "attribute_ruler", "tagger"]
    )
    if "senter" in nlp.component_names:
        nlp.enable_pipe("senter")
    else:  # model without a statistical senter — rule-based fallback
        nlp.add_pipe("sentencizer")
    return nlp


def _units(documents_path: Path, limit: int | None) -> Iterator[tuple[str, dict]]:
    """Yield (paragraph_text, context) pairs for spaCy's pipe.

    Segmentation runs **per paragraph**, not per section. Feeding whole sections in —
    paragraphs joined by ``\\n\\n`` — let the senter run straight through the blank line:
    9.1% of sentences came out containing a newline, merging the last sentence of one
    paragraph with the first of the next. That inflates the co-occurrence window and
    manufactures candidate pairs between people who are never in the same sentence.

    A paragraph break is an unambiguous sentence boundary, so it is applied directly
    rather than left to the model.
    """
    with documents_path.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if limit and index >= limit:
                return
            if not line.strip():
                continue
            doc = json.loads(line)
            for section_idx, section in enumerate(doc["sections"]):
                offset = 0
                for paragraph in section["text"].split("\n\n"):
                    if paragraph.strip():
                        yield paragraph, {
                            "doc_id": doc["doc_id"],
                            "section_idx": section_idx,
                            "para_offset": offset,
                        }
                    offset += len(paragraph) + 2  # the "\n\n" that was split away


def run(
    documents_path: Path,
    out_path: Path,
    *,
    limit: int | None = None,
    n_process: int = 1,
    batch_size: int = 200,
) -> dict[str, Any]:
    """Split every section into sentences with offsets. Returns counts."""
    nlp = load_senter()
    nlp.max_length = 2_000_000  # von Neumann's article is ~92k chars; headroom is cheap

    out_path.parent.mkdir(parents=True, exist_ok=True)
    counts = {"paragraphs": 0, "sentences": 0, "documents": 0, "sections": 0}
    seen_docs: set[str] = set()
    seen_sections: set[tuple[str, int]] = set()
    # sent_idx must run across the whole section, not restart per paragraph.
    section_counter: dict[tuple[str, int], int] = {}

    with out_path.open("w", encoding="utf-8", newline="\n") as out:
        stream = nlp.pipe(
            _units(documents_path, limit),
            as_tuples=True,
            batch_size=batch_size,
            n_process=n_process,
        )
        for doc, ctx in stream:
            counts["paragraphs"] += 1
            seen_docs.add(ctx["doc_id"])
            key = (ctx["doc_id"], ctx["section_idx"])
            seen_sections.add(key)
            for sent in doc.sents:
                text = sent.text.strip()
                if not text:
                    continue
                # Re-derive exact offsets: .strip() above must not desync start/end,
                # and offsets are relative to the *section*, so add the paragraph base.
                start = (
                    ctx["para_offset"]
                    + sent.start_char
                    + (len(sent.text) - len(sent.text.lstrip()))
                )
                sent_idx = section_counter.get(key, 0)
                section_counter[key] = sent_idx + 1
                out.write(
                    json.dumps(
                        {
                            "doc_id": ctx["doc_id"],
                            "section_idx": ctx["section_idx"],
                            "sent_idx": sent_idx,
                            "start": start,
                            "end": start + len(text),
                            "text": text,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                counts["sentences"] += 1
    counts["sections"] = len(seen_sections)

    counts["documents"] = len(seen_docs)
    write_manifest(
        out_path.parent,
        stage="segment",
        spec="0002-entity-mention-layer",
        config={"model": MODEL, "limit": limit, "n_process": n_process},
        counts=counts,
        inputs={"documents": documents_path},
    )
    return counts
