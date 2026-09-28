"""Layer version registry for `data/interim/`.

Each pipeline layer (corpus acquisition — spec 0001, entity & mention — spec 0002,
attribute spans — spec 0004, relations — spec 0005; `relation_typology` — spec 0003 —
kept for historical/ablation reruns, superseded for new work) writes into its own
`data/interim/{layer}/{version}/` directory instead of a shared flat one, so a rerun with
a different config never overwrites an existing evaluation snapshot and per-stage
manifests stop colliding.

The current version per layer is recorded in `data_versions.json` at the repo root and
incremented **by hand** — there is no automatic bump. A stage reads the version(s) of the
layer(s) it depends on to find its input, and its own layer's version to place its output.
"""

from __future__ import annotations

import json
from pathlib import Path

VERSIONS_FILE = Path("data_versions.json")
LAYERS = (
    "corpus_acquisition",
    "entity_mention_layer",
    "relation_typology",
    "attribute_spans",
    "relations",
)


def load_versions(path: Path = VERSIONS_FILE) -> dict[str, str]:
    """Read the layer -> version mapping, failing loudly if a layer is missing."""
    versions = json.loads(path.read_text(encoding="utf-8"))
    missing = [layer for layer in LAYERS if layer not in versions]
    if missing:
        raise KeyError(f"{path} is missing a version for: {', '.join(missing)}")
    return versions


def layer_dir(
    layer: str, *, base: Path = Path("data/interim"), path: Path = VERSIONS_FILE
) -> Path:
    """`base/{layer}/{current version}`, resolved from `data_versions.json`."""
    if layer not in LAYERS:
        raise ValueError(f"unknown layer {layer!r}, expected one of {LAYERS}")
    version = load_versions(path)[layer]
    return base / layer / version
