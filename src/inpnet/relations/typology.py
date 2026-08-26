"""The relation-typology plug-in contract. Implements spec 0003 §Decision 6.

A domain-specific relation typology (the physics-exemplary one is a follow-on spec's
job) is one YAML config: a list of relation types, each mapping onto one or more
clusters from a named `cluster` stage run. `load_typology` is meant to be the single
function an annotation-guide renderer, a model-prompt builder, and the evaluation script
all import, so the three cannot silently drift out of sync with each other.

Schema (see `docs/specs/typologies/TEMPLATE.yaml`):

    type_id: string               # stable short id, unique within the config
    label: string                 # human-readable name
    definition: string            # what an annotator checks for
    direction: directed | symmetric
    cluster_ids: [int]             # non-empty
    clustering_run_id: string      # which `cluster` run this mapping is valid against
    wikidata_properties: [string]  # optional P-ids, e.g. ["P26"]; [] if none apply
    examples: [string]             # illustrative sentences
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from ..manifest import read_jsonl

ALLOWED_DIRECTIONS = {"directed", "symmetric"}
_WIKIDATA_PROPERTY_RE = re.compile(r"^P\d+$")
_REQUIRED_FIELDS = (
    "type_id",
    "label",
    "definition",
    "direction",
    "cluster_ids",
    "clustering_run_id",
    "wikidata_properties",
    "examples",
)


class TypologyError(ValueError):
    """A typology config is malformed, or references clusters that don't exist."""


@dataclass(frozen=True)
class RelationType:
    type_id: str
    label: str
    definition: str
    direction: str
    cluster_ids: list[int] = field(default_factory=list)
    clustering_run_id: str = ""
    wikidata_properties: list[str] = field(default_factory=list)
    examples: list[str] = field(default_factory=list)


def _validate_entry(entry: dict, *, index: int) -> RelationType:
    if not isinstance(entry, dict):
        raise TypologyError(f"entry {index}: expected a mapping, got {type(entry).__name__}")

    missing = [f for f in _REQUIRED_FIELDS if f not in entry]
    if missing:
        raise TypologyError(f"entry {index}: missing field(s) {missing}")

    type_id = entry["type_id"]
    if not isinstance(type_id, str) or not type_id:
        raise TypologyError(f"entry {index}: 'type_id' must be a non-empty string")

    for str_field in ("label", "definition", "clustering_run_id"):
        value = entry[str_field]
        if not isinstance(value, str) or not value:
            raise TypologyError(f"'{type_id}': '{str_field}' must be a non-empty string")

    direction = entry["direction"]
    if direction not in ALLOWED_DIRECTIONS:
        raise TypologyError(
            f"'{type_id}': 'direction' must be one of {sorted(ALLOWED_DIRECTIONS)}, "
            f"got {direction!r}"
        )

    cluster_ids = entry["cluster_ids"]
    if not isinstance(cluster_ids, list) or not cluster_ids:
        raise TypologyError(f"'{type_id}': 'cluster_ids' must be a non-empty list")
    if not all(isinstance(c, int) and not isinstance(c, bool) for c in cluster_ids):
        raise TypologyError(f"'{type_id}': every 'cluster_ids' entry must be an int")

    wikidata_properties = entry["wikidata_properties"]
    if not isinstance(wikidata_properties, list):
        raise TypologyError(f"'{type_id}': 'wikidata_properties' must be a list")
    bad = [p for p in wikidata_properties if not _WIKIDATA_PROPERTY_RE.match(str(p))]
    if bad:
        raise TypologyError(f"'{type_id}': malformed Wikidata property id(s) {bad}")

    examples = entry["examples"]
    if not isinstance(examples, list) or not all(isinstance(e, str) for e in examples):
        raise TypologyError(f"'{type_id}': 'examples' must be a list of strings")

    return RelationType(
        type_id=type_id,
        label=entry["label"],
        definition=entry["definition"],
        direction=direction,
        cluster_ids=list(cluster_ids),
        clustering_run_id=entry["clustering_run_id"],
        wikidata_properties=list(wikidata_properties),
        examples=list(examples),
    )


def _clusters_by_run(cluster_summary_path: Path) -> dict[str, set[int]]:
    by_run: dict[str, set[int]] = {}
    for row in read_jsonl(cluster_summary_path):
        by_run.setdefault(row["clustering_run_id"], set()).add(int(row["cluster_id"]))
    return by_run


def load_typology(
    path: Path, *, cluster_summary_path: Path | None = None
) -> list[RelationType]:
    """Parse and validate a typology config.

    Schema validation always runs. If `cluster_summary_path` is given, every
    `cluster_ids` entry is additionally checked against that file's
    `(clustering_run_id, cluster_id)` pairs -- the check spec 0003's acceptance
    criteria calls for: a config referencing a cluster absent from its named run fails
    validation. Omit it to validate the schema alone, e.g. before a `cluster` run exists.
    """
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise TypologyError(f"{path}: expected a YAML list of relation types")

    types = [_validate_entry(entry, index=i) for i, entry in enumerate(raw)]

    seen_ids = set()
    for t in types:
        if t.type_id in seen_ids:
            raise TypologyError(f"duplicate type_id: '{t.type_id}'")
        seen_ids.add(t.type_id)

    if cluster_summary_path is not None:
        by_run = _clusters_by_run(cluster_summary_path)
        for t in types:
            valid_clusters = by_run.get(t.clustering_run_id)
            if valid_clusters is None:
                raise TypologyError(
                    f"'{t.type_id}': clustering_run_id '{t.clustering_run_id}' not found "
                    f"in {cluster_summary_path}"
                )
            unknown = [c for c in t.cluster_ids if c not in valid_clusters]
            if unknown:
                raise TypologyError(
                    f"'{t.type_id}': cluster_ids {unknown} do not exist in run "
                    f"'{t.clustering_run_id}' ({cluster_summary_path})"
                )

    return types
