"""Relation attribute span detection. Implements spec 0004.

Detects `time`/`place`/`institution` via spaCy NER and `action` via the shortest
dependency-parse path between a candidate pair's two participant tokens. Person/entity
detection is not redone here -- spec 0002's linked/subject mentions (already present in
`candidates.jsonl`'s own `head_span`/`tail_span`) are the only source, per spec 0004's
Non-goals.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from ..manifest import read_jsonl, write_jsonl, write_manifest
from .cluster import _name_variants, build_sentence_starts, dedup_sentences

SPEC = "0004-relation-attribute-detection"
DEFAULT_MODEL = "en_core_web_sm"

#: spaCy NER label -> this spec's attr_type (spec 0004 §Decision, "direct NER labels").
NER_ATTR_TYPES = {
    "DATE": "time",
    "GPE": "place",
    "LOC": "place",
    "FAC": "place",
    "ORG": "institution",
}

#: Wikidata gender QID (spec 0002's `entities.jsonl`) -> third-person pronouns to try
#: first for the pronoun-resolution fallback below. Unknown/other genders (a small tail
#: in this corpus -- ~30 of 32,546 entities) fall back to trying both sets.
MALE_PRONOUNS = ("he", "him", "his")
FEMALE_PRONOUNS = ("she", "her", "hers")
GENDER_PRONOUNS = {
    "Q6581097": MALE_PRONOUNS,
    "Q6581072": FEMALE_PRONOUNS,
}


def _pronoun_span(text: str, gender: str | None) -> tuple[int, int] | None:
    """Earliest occurrence of a gender-appropriate third-person pronoun in `text`.

    Diagnostic finding (2026-09-28): 95.6% of all `action`-detection misses (42.4% of
    *every* candidate pair) trace to exactly one cause -- a `subject_link` candidate
    whose subject is referred to only by pronoun in that sentence ("He did doctoral
    research under..."), never by name, so `locate_participant_span`'s name-match finds
    nothing. This is spec 0002's already-documented coreference gap (§Non-goals),
    previously unquantified.

    Accepted-for-now heuristic, deliberately simple: assume every third-person pronoun
    in the sentence refers to the article subject. **This is a known pitfall, not a
    validated resolution** -- a sentence can pronoun-reference someone other than the
    subject ("She raised her son, who later..." with a male subject would wrongly match
    "he"/"his" elsewhere in the sentence to the wrong person; here it can also pick a
    pronoun that actually refers to a third party). Every caller must treat a `"pronoun"`
    resolution method as lower-confidence than `"span"`/`"name"` and this fact must stay
    visible downstream (manifests, `relations.jsonl`'s `pronoun_resolved` flag) rather
    than silently folded in as equivalent -- see spec 0004 Risks.
    """
    candidates = GENDER_PRONOUNS.get(gender, MALE_PRONOUNS + FEMALE_PRONOUNS)
    earliest = None
    for pronoun in candidates:
        match = re.search(r"\b" + pronoun + r"\b", text, flags=re.IGNORECASE)
        if match and (earliest is None or match.start() < earliest.start()):
            earliest = match
    return earliest.span() if earliest else None


def load_detector_nlp(model: str = DEFAULT_MODEL):
    """spaCy with `parser`+`ner` enabled, everything else excluded.

    `tok2vec` is a shared dependency of both and loads automatically. `tagger`,
    `attribute_ruler`, `lemmatizer` and `senter` are excluded: sentences are already
    segmented (spec 0002 §Decision 4) and this spec reads neither POS tags nor lemmas.
    Also used tokenizer-only (`nlp.tokenizer(text)`, bypassing the pipeline) by
    `assemble.py`'s coverage computation, so the exact same token boundaries back both
    specs.
    """
    import spacy

    return spacy.load(model, exclude=["tagger", "attribute_ruler", "lemmatizer", "senter"])


def sentence_key(row: dict) -> str:
    return f"{row['doc_id']}:{row['section_idx']}:{row['sent_idx']}"


def locate_participant_span(
    text: str,
    entity_id: str,
    section_span: list[int] | None,
    sentence_start: int,
    canonical_names: dict[str, str],
    genders: dict[str, str] | None = None,
) -> tuple[int, int, str] | None:
    """Sentence-local `(start, end, method)` for one of a candidate pair's two
    participants. `method` is `"span"`, `"name"`, or `"pronoun"` (weakest -- see
    `_pronoun_span`), so a caller can tell a heuristic resolution from a verified one.

    A linked participant already carries a section-relative span directly on the
    `candidates.jsonl` row (spec 0002's `_candidate()`); converted to sentence-local by
    subtracting `sentence_start`, same conversion spec 0003 §2a used for masking
    (`method="span"`). The article subject carries no span at all (spec 0002
    §Decision 3 -- Wikipedia never self-links) and is located by name first, reusing
    spec 0003 §2a's own `_name_variants` matching (`method="name"`); if that also fails,
    falls back to `_pronoun_span` (`method="pronoun"`). Returns None only if none of the
    three resolve.
    """
    if section_span is not None:
        start, end = section_span[0] - sentence_start, section_span[1] - sentence_start
        if 0 <= start < end <= len(text):
            return start, end, "span"
        return None
    name = canonical_names.get(entity_id)
    if name:
        for variant in _name_variants(name):
            match = re.search(r"\b" + re.escape(variant) + r"\b", text)
            if match:
                return (*match.span(), "name")
    pronoun_span = _pronoun_span(text, (genders or {}).get(entity_id))
    if pronoun_span:
        return (*pronoun_span, "pronoun")
    return None


def shortest_dependency_span(
    doc: Any, head_char: tuple[int, int], tail_char: tuple[int, int]
) -> tuple[int, int] | None:
    """Minimal enclosing `(start, end)` covering the shortest dependency path between
    the two participants' root tokens -- spec 0004 §Decision's `action` mechanism.

    Each participant span's `.root` (the token in the span closest to the sentence
    root) stands in for "the participant" in the parse tree. The path is the two
    tokens' ancestor chains up to their lowest common ancestor; the returned span is
    the character range from the leftmost to the rightmost token on that path, so
    `value == text[span[0]:span[1]]` always holds for the caller -- some tokens inside
    that range may not themselves be on the path, but this keeps `value` and `span`
    consistent rather than reporting a not-contiguous "value". Returns None when either
    participant span doesn't resolve to a token, the two resolve to the same token, or
    no common ancestor is found (Span roots technically always share the sentence root
    in one parse, so the last case is defensive, not expected).
    """
    head_span = doc.char_span(*head_char, alignment_mode="expand")
    tail_span = doc.char_span(*tail_char, alignment_mode="expand")
    if head_span is None or tail_span is None:
        return None
    head_tok, tail_tok = head_span.root, tail_span.root
    if head_tok.i == tail_tok.i:
        return None

    def chain(tok):
        path = [tok]
        while path[-1].head.i != path[-1].i:
            path.append(path[-1].head)
        return path  # token -> ... -> sentence root

    head_chain, tail_chain = chain(head_tok), chain(tail_tok)
    tail_ancestor_idxs = {t.i for t in tail_chain}
    lca = next((t for t in head_chain if t.i in tail_ancestor_idxs), None)
    if lca is None:
        return None

    path_tokens = {t.i: t for t in head_chain[: head_chain.index(lca) + 1]}
    path_tokens.update({t.i: t for t in tail_chain[: tail_chain.index(lca) + 1]})
    starts = [t.idx for t in path_tokens.values()]
    ends = [t.idx + len(t.text) for t in path_tokens.values()]
    return min(starts), max(ends)


def detect_ner_attributes(doc: Any) -> list[dict]:
    """`time`/`place`/`institution` spans from one sentence's NER pass."""
    attrs = []
    for ent in doc.ents:
        attr_type = NER_ATTR_TYPES.get(ent.label_)
        if attr_type is None:
            continue
        attrs.append(
            {
                "attr_type": attr_type,
                "value": ent.text,
                "span": [ent.start_char, ent.end_char],
                "detector": "spacy_ner",
            }
        )
    return attrs


def run(
    candidates_path: Path,
    entities_path: Path,
    sentences_path: Path,
    out_dir: Path,
    *,
    model_name: str = DEFAULT_MODEL,
    limit: int | None = None,
    nlp: Any | None = None,
) -> dict[str, Any]:
    """Run the `detect-attributes` stage end to end. Implements spec 0004 §Interface.

    `nlp`, when given, replaces the real spaCy pipeline -- lets tests inject a small
    loaded pipeline instead of depending on `en_core_web_sm` being present, same pattern
    spec 0003's `embedder` parameter uses for sentence-transformers.
    """
    candidates = read_jsonl(candidates_path)
    if limit is not None:
        candidates = candidates[:limit]
    entities = read_jsonl(entities_path)
    sentences = read_jsonl(sentences_path)
    sentence_starts = build_sentence_starts(sentences)
    canonical_names = {e["qid"]: e["canonical_name"] for e in entities}
    genders = {e["qid"]: e.get("gender") for e in entities}

    unique_sentences = dedup_sentences(candidates)
    unique_sentences.sort(key=lambda r: (r["doc_id"], r["section_idx"], r["sent_idx"]))

    detector = nlp or load_detector_nlp(model_name)
    docs_by_key = {
        row["sentence_id"]: doc
        for row, doc in zip(
            unique_sentences,
            detector.pipe([row["sentence"] for row in unique_sentences], batch_size=200),
            strict=True,
        )
    }

    spans: list[dict] = []
    for key, doc in docs_by_key.items():
        for attr in detect_ner_attributes(doc):
            spans.append({"sentence_id": key, "candidate_id": None, **attr})

    n_action_found = n_action_missing = n_action_pronoun_resolved = 0
    for cand in candidates:
        key = sentence_key(cand)
        doc = docs_by_key.get(key)
        if doc is None:
            continue
        base = sentence_starts.get(key, 0)
        head = locate_participant_span(
            cand["sentence"], cand["head_entity_id"], cand.get("head_span"), base,
            canonical_names, genders,
        )
        tail = locate_participant_span(
            cand["sentence"], cand["tail_entity_id"], cand.get("tail_span"), base,
            canonical_names, genders,
        )
        action_span = None
        if head is not None and tail is not None:
            action_span = shortest_dependency_span(doc, head[:2], tail[:2])
        if action_span is None:
            n_action_missing += 1
            continue
        pronoun_used = head[2] == "pronoun" or tail[2] == "pronoun"
        if pronoun_used:
            n_action_pronoun_resolved += 1
        start, end = action_span
        spans.append(
            {
                "sentence_id": key,
                "candidate_id": cand["candidate_id"],
                "attr_type": "action",
                "value": cand["sentence"][start:end],
                "span": [start, end],
                "detector": "dep_parse",
                "pronoun_resolved": pronoun_used,
            }
        )
        n_action_found += 1

    write_jsonl(out_dir / "attribute_spans.jsonl", spans)

    spans_by_type: dict[str, int] = {}
    for s in spans:
        spans_by_type[s["attr_type"]] = spans_by_type.get(s["attr_type"], 0) + 1

    counts = {
        "n_candidates": len(candidates),
        "n_sentences": len(unique_sentences),
        "n_spans": len(spans),
        "spans_by_type": spans_by_type,
        "n_action_found": n_action_found,
        "n_action_missing": n_action_missing,
        "action_found_fraction": round(n_action_found / len(candidates), 4) if candidates else 0.0,
        # Pitfall tracker (see `_pronoun_span`): how much of `action_found` rests on the
        # "every pronoun is the subject" heuristic rather than a verified participant
        # location -- surfaced here so it stays visible, not folded into the headline
        # number as if it were equally reliable.
        "n_action_pronoun_resolved": n_action_pronoun_resolved,
        "action_pronoun_resolved_fraction": (
            round(n_action_pronoun_resolved / n_action_found, 4) if n_action_found else 0.0
        ),
    }
    write_manifest(
        out_dir,
        stage="detect-attributes",
        spec=SPEC,
        config={"model_name": model_name, "limit": limit},
        counts=counts,
        inputs={
            "candidates": candidates_path,
            "entities": entities_path,
            "sentences": sentences_path,
        },
    )
    return counts
