"""Run manifests. Implements the manifest requirement in spec 0001 §Interface.

Every stage writes `_manifest.json` next to its output so a result can be traced
back to the inputs, config and tool versions that produced it.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from . import __version__


def file_hash(path: Path, *, chunk: int = 1 << 20) -> str:
    """SHA-256 of a file, streamed so large snapshots don't land in memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def write_manifest(
    out_dir: Path,
    *,
    stage: str,
    config: dict[str, Any],
    counts: dict[str, Any],
    inputs: dict[str, Path] | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write `_manifest.json` into out_dir and return its path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "stage": stage,
        "spec": "0001-corpus-acquisition",
        "created_at": datetime.now(UTC).isoformat(),
        "inpnet_version": __version__,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "config": config,
        "counts": counts,
    }
    if inputs:
        manifest["inputs"] = {
            name: {
                "path": str(path),
                "sha256": file_hash(path) if path.is_file() else None,
            }
            for name, path in inputs.items()
        }
    if extra:
        manifest.update(extra)

    path = out_dir / "_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def read_jsonl(path: Path) -> list[dict]:
    """Read a JSONL file into a list of dicts, skipping blank lines."""
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    """Write rows as JSONL, UTF-8, one compact object per line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
