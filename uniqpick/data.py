"""Dataset discovery, pairing and duplicate grouping."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .captions import load_tags
from .features import dhash, file_md5, hamming, image_size

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

#: dhash values of images with no structure (flat colour, blank scans).
#: They match everything, so they must not drive near-duplicate merging.
_FLAT_DHASH = frozenset({0, (1 << 64) - 1})


@dataclass
class Record:
    """One image plus its metadata and captions."""

    path: Path
    width: int
    height: int
    md5: str
    dhash: int
    tags: dict[str, float] = field(default_factory=dict)
    duplicate_of: "Record | None" = None
    dup_reason: str | None = None

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def name(self) -> str:
        return self.path.name


def discover(
    roots: list[Path],
    recursive: bool = False,
    caption_suffixes: tuple[str, ...] = (".txt", ".personal.txt"),
    extra_exclude: tuple[str, ...] = (),
) -> list[Record]:
    """Find images under ``roots`` and load their hashes, size and captions."""
    files: list[Path] = []
    for root in roots:
        pattern = "**/*" if recursive else "*"
        files.extend(
            path
            for path in sorted(root.glob(pattern))
            if path.is_file() and path.suffix.lower() in IMAGE_EXTS
        )
    files = sorted(set(files))

    records: list[Record] = []
    for path in files:
        width, height = image_size(path)
        records.append(
            Record(
                path=path,
                width=width,
                height=height,
                md5=file_md5(path),
                dhash=dhash(path),
                tags=load_tags(path, caption_suffixes, extra_exclude=extra_exclude),
            )
        )
    return records


def dedupe(records: list[Record], max_hamming: int = 6) -> list[Record]:
    """Collapse exact and near duplicates, keeping the highest resolution copy.

    Returns the kept records.  Every dropped record gets ``duplicate_of`` and
    ``dup_reason`` set so the report can explain the cut.
    """
    parent = list(range(len(records)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    for i in range(len(records)):
        for j in range(i + 1, len(records)):
            a, b = records[i], records[j]
            if a.md5 == b.md5:
                union(i, j)
            elif (
                max_hamming >= 0
                and a.dhash not in _FLAT_DHASH
                and b.dhash not in _FLAT_DHASH
                and hamming(a.dhash, b.dhash) <= max_hamming
            ):
                union(i, j)

    groups: dict[int, list[int]] = {}
    for i in range(len(records)):
        groups.setdefault(find(i), []).append(i)

    kept: list[Record] = []
    for members in groups.values():
        # Highest resolution wins; ties keep the first one seen for stable output.
        members.sort(key=lambda idx: (-records[idx].area, idx))
        keeper = records[members[0]]
        kept.append(keeper)
        for idx in members[1:]:
            duplicate = records[idx]
            duplicate.duplicate_of = keeper
            duplicate.dup_reason = (
                "exact duplicate" if duplicate.md5 == keeper.md5 else "near duplicate"
            )

    # Preserve input order so reports and ranks stay predictable.
    order = {id(record): i for i, record in enumerate(records)}
    kept.sort(key=lambda record: order[id(record)])
    return kept
