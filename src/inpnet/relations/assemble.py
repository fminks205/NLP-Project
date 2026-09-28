"""Schemaless relation record assembly. Implements spec 0005.

Combines spec 0002's participants with spec 0004's detected attribute spans into one
relation record per candidate pair, measures how much of the sentence that record
actually accounts for, and splits low-coverage records out for review rather than
keeping or dropping them silently.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..manifest import read_jsonl, write_jsonl, write_manifest
from .attributes import load_detector_nlp, locate_participant_span, sentence_key
from .cluster import build_sentence_starts

SPEC = "0005-schemaless-relation-attributes"
DEFAULT_MIN_COVERAGE = 0.5
#: Merge content tokens separated by no more than this many stopword/punct tokens into
#: one uncovered run for display -- a readability choice for the human-readable output,
#: not part of the coverage number itself (see `_uncovered_runs`).
MAX_BRIDGE_TOKENS = 2


def _content_tokens(doc: Any) -> list[Any]:
    return [t for t in doc if not t.is_space and not t.is_stop and not t.is_punct]


def compute_coverage(doc: Any, covered_spans: list[tuple[int, int]]) -> float:
    """Fraction of content tokens (non-stopword, non-punct) inside any covered span.

    Character-based coverage would skew toward long spans regardless of relevance (spec
    0005 §Alternatives); token-based tracks how much of what the sentence actually
    asserts was captured.
    """
    tokens = _content_tokens(doc)
    if not tokens:
        return 1.0
    covered = sum(
        1 for t in tokens if any(s <= t.idx < e for s, e in covered_spans)
    )
    return covered / len(tokens)


def _uncovered_runs(doc: Any, text: str, covered_spans: list[tuple[int, int]]) -> list[str]:
    """Readable text runs a human should look at -- the content this record missed.

    Walks all tokens in order; a content token outside every covered span starts or
    extends a run. A short bridge of stopword/punct tokens (`MAX_BRIDGE_TOKENS`) between
    two uncovered content tokens is folded into the same run instead of splitting it, so
    "to A.E.R.E." reads as one run rather than three. Display-only: does not feed back
    into `compute_coverage`.
    """
    runs: list[tuple[int, int]] = []
    run_start = run_end = None
    bridge = 0
    for t in doc:
        if t.is_space:
            continue
        covered = any(s <= t.idx < e for s, e in covered_spans)
        content = not t.is_stop and not t.is_punct
        if content and not covered:
            if run_start is None:
                run_start = t.idx
            run_end = t.idx + len(t.text)
            bridge = 0
        elif run_start is not None and bridge < MAX_BRIDGE_TOKENS:
            bridge += 1
        else:
            if run_start is not None:
                runs.append((run_start, run_end))
            run_start = run_end = None
            bridge = 0
    if run_start is not None:
        runs.append((run_start, run_end))
    return [text[s:e] for s, e in runs]


def build_relation_record(
    candidate: dict,
    doc: Any,
    sentence_attrs: list[dict],
    candidate_attrs: list[dict],
    head: tuple[int, int, str] | None,
    tail: tuple[int, int, str] | None,
) -> dict:
    """One relation record (spec 0005 §Interface schema) for a single candidate pair.

    `head`/`tail` are `locate_participant_span`'s `(start, end, method)` results (or
    None). `pronoun_resolved` on the returned record is true when either participant
    only resolved via the pronoun heuristic (`attributes.py`'s `_pronoun_span`) --
    surfaced per-record, not just as an aggregate manifest count, so a reviewer filtering
    `relations.jsonl` can find exactly which records rest on that heuristic.
    """
    attrs = sentence_attrs + candidate_attrs
    covered_spans = [tuple(a["span"]) for a in attrs]
    participants = []
    pronoun_resolved = False
    for entity_id, resolved in (
        (candidate["head_entity_id"], head),
        (candidate["tail_entity_id"], tail),
    ):
        participants.append({"mention_id": entity_id, "qid": entity_id})
        if resolved is not None:
            covered_spans.append(resolved[:2])
            pronoun_resolved = pronoun_resolved or resolved[2] == "pronoun"

    coverage = compute_coverage(doc, covered_spans)
    uncovered = _uncovered_runs(doc, candidate["sentence"], covered_spans)

    return {
        "relation_id": candidate["candidate_id"],
        "sentence_id": sentence_key(candidate),
        "sentence": candidate["sentence"],
        "participants": participants,
        "attributes": [
            {
                "attr_type": a["attr_type"],
                "value": a["value"],
                "span": a["span"],
                "source": a["detector"],
            }
            for a in attrs
        ],
        "coverage": round(coverage, 4),
        "uncovered": uncovered,  # display-only; not part of spec 0005's core schema
        "pronoun_resolved": pronoun_resolved,
    }


def format_relations_summary(rows: list[dict]) -> str:
    """Render relation records for human review -- spec 0005's human-readable output.

    One block per record: the sentence, its typed attributes, and (per this spec's
    Goal) whatever content `coverage` did not count as covered, spelled out rather than
    left implicit in the number. Serves both `relations.jsonl` and
    `low_coverage_relations.jsonl` -- same schema, same renderer.
    """
    if not rows:
        return "(no relations)"
    blocks = []
    for r in rows:
        flag = "  [pronoun-resolved participant -- heuristic, unverified]" if r.get("pronoun_resolved") else ""
        lines = [f"relation {r['relation_id']}   coverage {r['coverage']:.2f}{flag}"]
        lines.append(f"    sentence: {r['sentence']}")
        participants = ", ".join(p["qid"] for p in r["participants"])
        lines.append(f"    participants: {participants}")
        if r["attributes"]:
            lines.append("    attributes:")
            for a in r["attributes"]:
                lines.append(f"      {a['attr_type']}: {a['value']!r} ({a['source']})")
        else:
            lines.append("    attributes: (none detected)")
        uncovered = r.get("uncovered") or []
        if uncovered:
            lines.append("    unclassified:")
            for u in uncovered:
                lines.append(f"      - {u!r}")
        else:
            lines.append("    unclassified: (none)")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def run(
    candidates_path: Path,
    entities_path: Path,
    sentences_path: Path,
    attributes_path: Path,
    out_dir: Path,
    *,
    min_coverage: float = DEFAULT_MIN_COVERAGE,
    model_name: str = "en_core_web_sm",
    limit: int | None = None,
    nlp: Any | None = None,
) -> dict[str, Any]:
    """Run the `assemble-relations` stage end to end. Implements spec 0005 §Interface."""
    candidates = read_jsonl(candidates_path)
    if limit is not None:
        candidates = candidates[:limit]
    entities = read_jsonl(entities_path)
    sentences = read_jsonl(sentences_path)
    attribute_spans = read_jsonl(attributes_path)

    canonical_names = {e["qid"]: e["canonical_name"] for e in entities}
    genders = {e["qid"]: e.get("gender") for e in entities}
    sentence_starts = build_sentence_starts(sentences)

    sentence_attrs: dict[str, list[dict]] = {}
    candidate_attrs: dict[str, list[dict]] = {}
    for row in attribute_spans:
        if row.get("candidate_id"):
            candidate_attrs.setdefault(row["candidate_id"], []).append(row)
        else:
            sentence_attrs.setdefault(row["sentence_id"], []).append(row)

    detector = nlp or load_detector_nlp(model_name)

    relations: list[dict] = []
    low_coverage: list[dict] = []
    for cand in candidates:
        key = sentence_key(cand)
        doc = detector.tokenizer(cand["sentence"])
        base = sentence_starts.get(key, 0)
        head = locate_participant_span(
            cand["sentence"], cand["head_entity_id"], cand.get("head_span"), base,
            canonical_names, genders,
        )
        tail = locate_participant_span(
            cand["sentence"], cand["tail_entity_id"], cand.get("tail_span"), base,
            canonical_names, genders,
        )
        record = build_relation_record(
            cand,
            doc,
            sentence_attrs.get(key, []),
            candidate_attrs.get(cand["candidate_id"], []),
            head,
            tail,
        )
        (relations if record["coverage"] >= min_coverage else low_coverage).append(record)

    write_jsonl(out_dir / "relations.jsonl", relations)
    write_jsonl(out_dir / "low_coverage_relations.jsonl", low_coverage)
    (out_dir / "relations_summary.txt").write_text(
        format_relations_summary(relations), encoding="utf-8"
    )
    (out_dir / "low_coverage_relations_summary.txt").write_text(
        format_relations_summary(low_coverage), encoding="utf-8"
    )

    total = len(relations) + len(low_coverage)
    all_records = relations + low_coverage
    coverages = [r["coverage"] for r in all_records]
    n_pronoun_resolved = sum(1 for r in all_records if r["pronoun_resolved"])
    counts = {
        "n_candidates": len(candidates),
        "n_relations": len(relations),
        "n_low_coverage": len(low_coverage),
        "low_coverage_fraction": round(len(low_coverage) / total, 4) if total else 0.0,
        "min_coverage": min_coverage,
        "mean_coverage": round(sum(coverages) / total, 4) if total else 0.0,
        # Pitfall tracker (see attributes.py's `_pronoun_span`) -- how many records rest
        # on the "every pronoun is the subject" heuristic, surfaced rather than hidden.
        "n_pronoun_resolved": n_pronoun_resolved,
        "pronoun_resolved_fraction": round(n_pronoun_resolved / total, 4) if total else 0.0,
    }
    write_manifest(
        out_dir,
        stage="assemble-relations",
        spec=SPEC,
        config={"min_coverage": min_coverage, "model_name": model_name, "limit": limit},
        counts=counts,
        inputs={
            "candidates": candidates_path,
            "entities": entities_path,
            "sentences": sentences_path,
            "attribute_spans": attributes_path,
        },
    )
    return counts
