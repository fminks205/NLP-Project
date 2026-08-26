"""Build the mention layer and candidate pairs. Implements spec 0002 §Decision 3 and 5.

Two mention sources, both model-free:

* **link** — a wiki link whose target resolves to a Wikidata human. Human-curated, so
  precision here is as good as Wikipedia's own annotation.
* **subject** — the article's own subject. A biography almost never links itself, yet its
  subject is a party to most relations the article asserts. Recorded as its own
  `mention_type` so the heuristic can be ablated rather than silently baked in.
"""

from __future__ import annotations

import bisect
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from ..manifest import read_jsonl, write_jsonl, write_manifest
from .resolve import collect_link_targets

RULE_SUBJECT_LINK = "subject_link"
RULE_LINK_LINK = "link_link"

#: Above this many distinct people in one sentence, link_link pairs are suppressed.
#:
#: Enumerations are the dominant failure mode of the link_link rule. "The most prominent
#: physicists participated: Bohr, Dirac, Yukawa, Schwinger, Salam, Prigogine..." yields
#: C(n,2) pairs, and none of them assert a relation *between* the listed people — each
#: relates to the article's subject. Measured before capping: 61% of link_link pairs came
#: from sentences listing 4+ people, and one sentence alone produced 325 pairs.
#:
#: subject_link is deliberately NOT capped: in exactly those sentences, subject-to-each
#: is the relation the text actually asserts.
MAX_PEOPLE_FOR_LINK_LINK = 6


class IncompleteCache(RuntimeError):
    """The Wikidata cache does not cover the corpus's link targets."""


def load_human_index(cache_path: Path) -> dict[str, dict]:
    """Map link-target title -> Wikidata record, keeping only humans."""
    return {
        row["title"]: row
        for row in read_jsonl(cache_path)
        if row.get("is_human") and row.get("qid")
    }


def check_cache_coverage(
    documents_path: Path, cache_path: Path, *, strict: bool = True
) -> dict[str, Any]:
    """Verify the cache covers every link target in the corpus.

    Without this, a short cache is indistinguishable from a corpus full of non-people:
    unresolved titles simply produce no mentions, the run succeeds, and the graph comes
    out near-empty. Failing loudly is the difference between a bug and a silent result.

    Raises:
        IncompleteCache: when `strict` and any target is absent or has
            `claims_resolved` false.
    """
    targets = set(collect_link_targets(documents_path))
    cache = {row["title"]: row for row in read_jsonl(cache_path)}

    absent = sorted(targets - set(cache))
    unresolved = sorted(
        title
        for title in targets & set(cache)
        # Records written before claims_resolved existed are treated as resolved.
        if cache[title].get("claims_resolved", True) is False
    )
    report = {
        "link_targets": len(targets),
        "cached": len(cache),
        "absent_from_cache": len(absent),
        "claims_unresolved": len(unresolved),
        "examples": (absent[:3] + unresolved[:3])[:5],
    }
    if absent or unresolved:
        message = (
            f"Wikidata cache covers {len(targets) - len(absent) - len(unresolved):,} "
            f"of {len(targets):,} link targets — {len(absent):,} absent, "
            f"{len(unresolved):,} unresolved. Examples: {report['examples']}. "
            f"Run `inpnet resolve` to completion, or pass --lenient to proceed anyway "
            f"(the graph will silently omit these people)."
        )
        if strict:
            raise IncompleteCache(message)
        print(f"WARNING: {message}", flush=True)
    return report


def _sentence_index(sentences_path: Path) -> dict[tuple[str, int], dict]:
    """Index sentences by (doc_id, section_idx) with sorted starts for bisect lookup."""
    grouped: dict[tuple[str, int], list[dict]] = defaultdict(list)
    with sentences_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            grouped[(row["doc_id"], row["section_idx"])].append(row)

    index: dict[tuple[str, int], dict] = {}
    for key, rows in grouped.items():
        rows.sort(key=lambda r: r["start"])
        index[key] = {"starts": [r["start"] for r in rows], "rows": rows}
    return index


def locate_sentence(index: dict, key: tuple[str, int], offset: int) -> dict | None:
    """Find the sentence containing `offset`, or None if it falls in a gap."""
    entry = index.get(key)
    if not entry:
        return None
    position = bisect.bisect_right(entry["starts"], offset) - 1
    if position < 0:
        return None
    row = entry["rows"][position]
    return row if row["start"] <= offset < row["end"] else None


def run(
    documents_path: Path,
    sentences_path: Path,
    cache_path: Path,
    seed_path: Path,
    out_dir: Path,
    *,
    limit: int | None = None,
    strict: bool = True,
    max_people_for_link_link: int = MAX_PEOPLE_FOR_LINK_LINK,
) -> dict[str, Any]:
    """Emit mentions, entities and candidate pairs for the corpus."""
    # A short cache produces a near-empty graph without erroring, so check first.
    coverage = check_cache_coverage(
        documents_path, cache_path, strict=strict and limit is None
    )
    humans = load_human_index(cache_path)
    seed = {p["qid"]: p for p in read_jsonl(seed_path)}
    sentences = _sentence_index(sentences_path)

    mentions: list[dict] = []
    candidates: list[dict] = []
    entity_docs: dict[str, set[str]] = defaultdict(set)
    entity_counts: dict[str, int] = defaultdict(int)
    entity_meta: dict[str, dict] = {}
    stats = {
        "documents": 0,
        "docs_without_pairs": 0,
        "link_mentions": 0,
        "subject_mentions": 0,
        "non_person_links": 0,   # target resolved, but is not a human
        "links_outside_sentence": 0,
        RULE_SUBJECT_LINK: 0,
        RULE_LINK_LINK: 0,
        "link_link_suppressed": 0,
    }

    with documents_path.open(encoding="utf-8") as handle:
        for doc_index, line in enumerate(handle):
            if limit and doc_index >= limit:
                break
            if not line.strip():
                continue
            doc = json.loads(line)
            doc_id = doc["doc_id"]
            subject_qid = doc["qid"] or doc_id
            stats["documents"] += 1

            # The subject is a document-level entity: no span until coreference exists.
            mentions.append(
                {
                    "mention_id": f"{doc_id}:subject",
                    "doc_id": doc_id,
                    "section_idx": None,
                    "sent_idx": None,
                    "start": None,
                    "end": None,
                    "surface": doc["title"],
                    "entity_id": subject_qid,
                    "mention_type": "subject",
                    "link_method": "subject",
                    "confidence": 1.0,
                }
            )
            stats["subject_mentions"] += 1
            entity_docs[subject_qid].add(doc_id)
            entity_counts[subject_qid] += 1
            entity_meta.setdefault(subject_qid, {"name": doc["title"], "wd": None})

            pairs_in_doc = 0
            # sentence key -> person mentions found in it
            per_sentence: dict[tuple[int, int], list[dict]] = defaultdict(list)

            for section_idx, section in enumerate(doc["sections"]):
                for link in section["links"]:
                    record = humans.get(link["target_title"])
                    if not record:
                        stats["non_person_links"] += 1
                        continue
                    sentence = locate_sentence(
                        sentences, (doc_id, section_idx), link["start"]
                    )
                    if sentence is None:
                        stats["links_outside_sentence"] += 1
                        continue

                    qid = record["qid"]
                    mention = {
                        "mention_id": f"{doc_id}:{section_idx}:{link['start']}",
                        "doc_id": doc_id,
                        "section_idx": section_idx,
                        "sent_idx": sentence["sent_idx"],
                        "start": link["start"],
                        "end": link["end"],
                        "surface": link["surface"],
                        "entity_id": qid,
                        "mention_type": "link",
                        "link_method": "wikilink",
                        "confidence": 1.0,
                    }
                    mentions.append(mention)
                    stats["link_mentions"] += 1
                    entity_docs[qid].add(doc_id)
                    entity_counts[qid] += 1
                    entity_meta.setdefault(
                        qid, {"name": record["resolved_title"] or record["title"], "wd": record}
                    )
                    per_sentence[(section_idx, sentence["sent_idx"])].append(
                        {"mention": mention, "sentence": sentence}
                    )

            # -- candidate pairs, scoped to a sentence -----------------
            for (section_idx, sent_idx), found in sorted(per_sentence.items()):
                sentence_text = found[0]["sentence"]["text"]
                entities_here = [f["mention"] for f in found]
                n_people = len({m["entity_id"] for m in entities_here} | {subject_qid})

                for mention in entities_here:
                    if mention["entity_id"] == subject_qid:
                        continue
                    candidates.append(
                        _candidate(
                            doc_id, section_idx, sent_idx, sentence_text,
                            subject_qid, mention["entity_id"],
                            None, mention, RULE_SUBJECT_LINK, n_people,
                        )
                    )
                    stats[RULE_SUBJECT_LINK] += 1
                    pairs_in_doc += 1

                if n_people > max_people_for_link_link:
                    # An enumeration: the listed people relate to the subject, not to
                    # each other. Counted so the suppression is visible, not silent.
                    distinct = len({m["entity_id"] for m in entities_here})
                    stats["link_link_suppressed"] += distinct * (distinct - 1) // 2
                    continue

                for i in range(len(entities_here)):
                    for j in range(i + 1, len(entities_here)):
                        a, b = entities_here[i], entities_here[j]
                        if a["entity_id"] == b["entity_id"]:
                            continue
                        candidates.append(
                            _candidate(
                                doc_id, section_idx, sent_idx, sentence_text,
                                a["entity_id"], b["entity_id"], a, b,
                                RULE_LINK_LINK, n_people,
                            )
                        )
                        stats[RULE_LINK_LINK] += 1
                        pairs_in_doc += 1

            if pairs_in_doc == 0:
                stats["docs_without_pairs"] += 1

    entities = [
        {
            "qid": qid,
            "canonical_name": meta["name"],
            "in_seed": qid in seed,
            "mention_count": entity_counts[qid],
            "doc_count": len(entity_docs[qid]),
            "doc_ids": sorted(entity_docs[qid])[:50],
            "birth_year": _meta(meta, seed, qid, "birth_year"),
            "death_year": _meta(meta, seed, qid, "death_year"),
            "gender": _meta(meta, seed, qid, "gender"),
            "occupations": (meta["wd"] or {}).get("occupations", []),
        }
        for qid, meta in sorted(entity_meta.items())
    ]

    write_jsonl(out_dir / "mentions.jsonl", mentions)
    write_jsonl(out_dir / "entities.jsonl", entities)
    write_jsonl(out_dir / "candidates.jsonl", candidates)

    counts = stats | {
        "entities": len(entities),
        "entities_in_seed": sum(1 for e in entities if e["in_seed"]),
        "candidates": len(candidates),
    }
    write_manifest(
        out_dir,
        stage="mentions",
        config={
            "limit": limit,
            "strict": strict,
            "max_people_for_link_link": max_people_for_link_link,
        },
        counts=counts,
        inputs={"documents": documents_path, "sentences": sentences_path},
        extra={"cache_coverage": coverage},
    )
    return counts


def _candidate(
    doc_id, section_idx, sent_idx, sentence, head, tail, head_m, tail_m, rule,
    people_in_sentence,
) -> dict:
    return {
        "candidate_id": f"{doc_id}:{section_idx}:{sent_idx}:{head}:{tail}",
        "doc_id": doc_id,
        "section_idx": section_idx,
        "sent_idx": sent_idx,
        "sentence": sentence,
        "head_entity_id": head,
        "tail_entity_id": tail,
        "head_span": [head_m["start"], head_m["end"]] if head_m else None,
        "tail_span": [tail_m["start"], tail_m["end"]] if tail_m else None,
        "rule": rule,
        "people_in_sentence": people_in_sentence,
    }


def _meta(meta: dict, seed: dict, qid: str, field: str):
    """Prefer seed metadata (already verified) and fall back to the resolver's."""
    if qid in seed and seed[qid].get(field) is not None:
        return seed[qid][field]
    return (meta["wd"] or {}).get(field)
