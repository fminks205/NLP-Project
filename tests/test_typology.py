"""Tests for the relation-typology plug-in contract. Implements spec 0003 §Decision 6.

Pure Python + PyYAML (a core dependency) -- no ML extras needed, so this suite always
runs. Uses a toy typology, not the physics-exemplary one, since that is a follow-on
spec's job (spec 0003 §Non-goals).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from inpnet.relations.typology import TypologyError, load_typology

VALID_ENTRY = {
    "type_id": "mentorship",
    "label": "Mentorship",
    "definition": "One person supervised or advised the other's academic work.",
    "direction": "directed",
    "cluster_ids": [0, 2],
    "clustering_run_id": "run123",
    "wikidata_properties": ["P184"],
    "examples": ["His PhD advisor was Fredrik Zachariasen."],
}


def _write_typology(tmp_path, entries, name="typology.yaml"):
    path = tmp_path / name
    path.write_text(yaml.safe_dump(entries), encoding="utf-8")
    return path


def _write_cluster_summary(tmp_path, rows):
    path = tmp_path / "cluster_summary.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    return path


def test_valid_typology_loads_from_file(tmp_path):
    path = _write_typology(tmp_path, [VALID_ENTRY])
    types = load_typology(path)
    assert len(types) == 1
    assert types[0].type_id == "mentorship"
    assert types[0].direction == "directed"
    assert types[0].wikidata_properties == ["P184"]


@pytest.mark.parametrize(
    "missing_field", ["type_id", "label", "definition", "direction", "cluster_ids"]
)
def test_missing_required_field_rejected(tmp_path, missing_field):
    entry = {k: v for k, v in VALID_ENTRY.items() if k != missing_field}
    path = _write_typology(tmp_path, [entry])
    with pytest.raises(TypologyError, match=missing_field):
        load_typology(path)


def test_bad_direction_rejected(tmp_path):
    entry = {**VALID_ENTRY, "direction": "sideways"}
    path = _write_typology(tmp_path, [entry])
    with pytest.raises(TypologyError, match="direction"):
        load_typology(path)


def test_empty_cluster_ids_rejected(tmp_path):
    entry = {**VALID_ENTRY, "cluster_ids": []}
    path = _write_typology(tmp_path, [entry])
    with pytest.raises(TypologyError, match="cluster_ids"):
        load_typology(path)


def test_malformed_wikidata_property_rejected(tmp_path):
    entry = {**VALID_ENTRY, "wikidata_properties": ["spouse"]}
    path = _write_typology(tmp_path, [entry])
    with pytest.raises(TypologyError, match="Wikidata property"):
        load_typology(path)


def test_duplicate_type_id_rejected(tmp_path):
    path = _write_typology(tmp_path, [VALID_ENTRY, VALID_ENTRY])
    with pytest.raises(TypologyError, match="duplicate"):
        load_typology(path)


def test_cluster_ids_validated_against_real_run(tmp_path):
    """Acceptance criterion: a config referencing a cluster absent from its named run fails."""
    summary = _write_cluster_summary(
        tmp_path,
        [
            {"cluster_id": 0, "clustering_run_id": "run123", "size": 5},
            {"cluster_id": 2, "clustering_run_id": "run123", "size": 3},
        ],
    )
    ok_path = _write_typology(tmp_path, [VALID_ENTRY], name="ok.yaml")
    load_typology(ok_path, cluster_summary_path=summary)  # does not raise

    bad_entry = {**VALID_ENTRY, "cluster_ids": [0, 99]}
    bad_path = _write_typology(tmp_path, [bad_entry], name="bad.yaml")
    with pytest.raises(TypologyError, match="99"):
        load_typology(bad_path, cluster_summary_path=summary)


def test_unknown_clustering_run_id_rejected(tmp_path):
    summary = _write_cluster_summary(
        tmp_path, [{"cluster_id": 0, "clustering_run_id": "some-other-run", "size": 5}]
    )
    path = _write_typology(tmp_path, [VALID_ENTRY])
    with pytest.raises(TypologyError, match="run123"):
        load_typology(path, cluster_summary_path=summary)


def test_template_file_is_a_valid_starting_point(tmp_path):
    """The checked-in TEMPLATE.yaml parses to one entry that only fails on the
    placeholder cluster/run id -- i.e. the schema itself is right."""
    template = Path("docs/specs/typologies/TEMPLATE.yaml")
    types = load_typology(template)
    assert len(types) == 1
    assert types[0].clustering_run_id == "REPLACE_ME"
