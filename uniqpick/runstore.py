"""Persisted run state: per-signal distance matrices plus the metadata to re-pick.

Computing the distances is the expensive half of uniqpick (hashing, caption
parsing, image embeddings).  Picking N of them out of the matrix is a greedy
walk that costs a few milliseconds, so a run folder stores everything the walk
needs and lets `uniqpick pick` reuse it for any k, method or signal weight.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .data import Record

META_NAME = "meta.json"
DIST_NAME = "distances.npz"
RUN_VERSION = 1


class RunFormatError(ValueError):
    """The folder is not a uniqpick run, or it is from an incompatible version."""


@dataclass
class RunData:
    """One computed run: the metadata and the pairwise distances it produced."""

    meta: dict[str, Any]
    records: list[Record]
    kept: list[Record]
    tag_dist: np.ndarray | None
    image_dist: np.ndarray | None
    aspect_dist: np.ndarray | None


def encode_records(records: list[Record]) -> list[dict[str, Any]]:
    """Plain-JSON records; dhash is dropped because nothing after compute uses it."""
    return [
        {
            "path": str(record.path),
            "width": record.width,
            "height": record.height,
            "md5": record.md5,
            "tags": {tag: round(float(weight), 4) for tag, weight in record.tags.items()},
            "duplicate_of": str(record.duplicate_of.path) if record.duplicate_of is not None else None,
            "dup_reason": record.dup_reason,
        }
        for record in records
    ]


def decode_records(entries: list[dict[str, Any]]) -> list[Record]:
    """Rebuild records; duplicate links are resolved through the recorded paths."""
    records = [
        Record(
            path=Path(entry["path"]),
            width=int(entry["width"]),
            height=int(entry["height"]),
            md5=str(entry["md5"]),
            dhash=0,
            tags={tag: float(weight) for tag, weight in entry.get("tags", {}).items()},
        )
        for entry in entries
    ]
    by_path = {str(record.path): record for record in records}
    for record, entry in zip(records, entries):
        keeper = entry.get("duplicate_of")
        if keeper is not None:
            record.duplicate_of = by_path.get(str(keeper))
            record.dup_reason = entry.get("dup_reason")
    return records


def save_run(
    run_dir: Path,
    meta: dict[str, Any],
    tag_dist: np.ndarray | None,
    image_dist: np.ndarray | None,
    aspect_dist: np.ndarray | None,
) -> Path:
    """Write ``meta.json`` and ``distances.npz`` into ``run_dir``."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    arrays = {
        name: np.asarray(matrix, dtype=np.float32)
        for name, matrix in (
            ("tag", tag_dist),
            ("image", image_dist),
            ("aspect", aspect_dist),
        )
        if matrix is not None
    }
    # np.savez appends .npz only when the name lacks it, and here the name has it.
    np.savez_compressed(run_dir / DIST_NAME, **arrays)
    (run_dir / META_NAME).write_text(
        json.dumps(meta, indent=1, ensure_ascii=False, sort_keys=False),
        encoding="utf-8",
    )
    return run_dir


def load_run(run_dir: Path) -> RunData:
    """Read a run folder written by :func:`save_run`."""
    run_dir = Path(run_dir)
    meta_path = run_dir / META_NAME
    if not meta_path.is_file():
        raise RunFormatError(
            f"{run_dir} is not a uniqpick run (no {META_NAME}); run `uniqpick compute` first"
        )

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    version = meta.get("version")
    if version != RUN_VERSION:
        raise RunFormatError(
            f"{run_dir} was written by run format {version}, this build reads {RUN_VERSION}"
        )

    records = decode_records(meta.get("records", []))
    kept_indices = [int(index) for index in meta.get("kept", [])]
    if not kept_indices:
        raise RunFormatError(f"{run_dir}/{META_NAME} has no kept records")
    kept = [records[index] for index in kept_indices]

    matrices: dict[str, np.ndarray] = {}
    dist_path = run_dir / DIST_NAME
    if dist_path.is_file():
        with np.load(dist_path, allow_pickle=False) as archive:
            matrices = {name: archive[name] for name in archive.files}

    for name, matrix in matrices.items():
        if matrix.shape != (len(kept), len(kept)):
            raise RunFormatError(
                f"{dist_path}: {name} matrix is {matrix.shape}, expected {len(kept)}x{len(kept)}"
            )
        if not np.allclose(matrix, matrix.T, atol=1e-5) or np.any(np.diag(matrix) > 1e-4):
            raise RunFormatError(f"{dist_path}: {name} matrix is not a zero-diagonal symmetric distance")

    return RunData(
        meta=meta,
        records=records,
        kept=kept,
        tag_dist=matrices.get("tag"),
        image_dist=matrices.get("image"),
        aspect_dist=matrices.get("aspect"),
    )
