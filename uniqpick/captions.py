"""Booru-style caption parsing and tag loading.

Captions are comma separated tag lists, optionally with weights in the usual
kohya / booru forms: ``(tag:1.2)``, ``[tag]`` and ``{tag}``.  Escaped commas
(``\\,``) and parentheses are respected so ``star (symbol)`` survives intact.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

# Tags that describe the file or the tagger rather than the picture.  These are
# dropped before scoring: they are either constant across the set or irrelevant
# to visual variety.
META_TAGS = frozenset(
    {
        "general",
        "sensitive",
        "safe",
        "questionable",
        "explicit",
        "nsfw",
        "high resolution",
        "absurd resolution",
        "incredibly absurd res",
        "huge filesize",
        "lowres",
        "bad quality",
        "best quality",
        "high quality",
        "normal quality",
        "worst quality",
        "masterpiece",
        "jpeg artifacts",
        "watermark",
        "different zoom layer",
    }
)

#: Prefixed metadata that varies by file rather than by picture, e.g. booru
#: ``rating:safe``.  Matched as a prefix because the value changes.
META_TAG_PREFIXES = ("rating:", "meta:", "spoiler:")

_WEIGHT_RE = re.compile(r"^\((.*):\s*([0-9]*\.?[0-9]+)\)$", re.DOTALL)
_ESCAPE_RE = re.compile(r"\\(.)")
_LITERAL_UNDERSCORE_RE = re.compile(r"(?<!\\)_")
_SPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class Tag:
    """One normalized tag with its (possibly bracketed) weight."""

    name: str
    weight: float = 1.0


def split_tags(text: str) -> list[str]:
    """Split a caption on top-level commas, keeping bracket groups intact."""
    parts: list[str] = []
    current: list[str] = []
    depth = 0
    i = 0
    while i < len(text):
        ch = text[i]
        if ch == "\\" and i + 1 < len(text):
            # Keep the escape so parse_tag can decide what it means, but do not
            # let the escaped character change bracket depth.
            current.append(ch)
            current.append(text[i + 1])
            i += 2
            continue
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth = max(0, depth - 1)
        if ch == "," and depth == 0:
            parts.append("".join(current))
            current = []
            i += 1
            continue
        current.append(ch)
        i += 1
    parts.append("".join(current))
    return [p.strip() for p in parts if p.strip()]


def parse_tag(raw: str) -> Tag | None:
    """Normalize one raw tag, unwrapping weight syntax.  Returns None if empty."""
    text = raw.strip()
    if not text:
        return None

    weight = 1.0
    match = _WEIGHT_RE.match(text)
    if match:
        text = match.group(1).strip()
        try:
            weight = float(match.group(2))
        except ValueError:
            weight = 1.0
    elif text.startswith("[") and text.endswith("]"):
        text, weight = text[1:-1].strip(), 0.95
    elif text.startswith("{") and text.endswith("}"):
        text, weight = text[1:-1].strip(), 1.05

    # Unescaped underscores are spaces; an escaped one is a literal underscore.
    text = _LITERAL_UNDERSCORE_RE.sub(" ", text)
    text = _ESCAPE_RE.sub(r"\1", text).strip()
    text = _SPACE_RE.sub(" ", text).lower()
    if not text:
        return None
    return Tag(text, weight)


def caption_paths(image: Path, suffixes: Iterable[str]) -> list[Path]:
    """Sidecar paths for an image, e.g. ``abc.jpg`` -> ``abc.txt``, ``abc.personal.txt``."""
    return [image.parent / (image.stem + suffix) for suffix in suffixes]


def load_tags(
    image: Path,
    suffixes: Iterable[str] = (".txt", ".personal.txt"),
    exclude: frozenset[str] = META_TAGS,
    extra_exclude: Iterable[str] = (),
) -> dict[str, float]:
    """Load and merge every sidecar caption for one image into ``{tag: weight}``.

    Duplicate tags keep the largest weight so a ``(tag:1.2)`` in one sidecar is
    not lost to a bare mention in another.
    """
    banned = set(exclude) | {t.lower() for t in extra_exclude}
    tags: dict[str, float] = {}
    for path in caption_paths(image, suffixes):
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for raw in split_tags(text):
            tag = parse_tag(raw)
            if tag is None or tag.name in banned:
                continue
            if tag.name.startswith(META_TAG_PREFIXES):
                continue
            if tag.weight > tags.get(tag.name, 0.0):
                tags[tag.name] = tag.weight
    return tags


def tag_frequencies(docs: Iterable[Mapping[str, float]]) -> dict[str, int]:
    """Document frequency for every tag across the corpus."""
    freq: dict[str, int] = {}
    for doc in docs:
        for tag in doc:
            freq[tag] = freq.get(tag, 0) + 1
    return freq
