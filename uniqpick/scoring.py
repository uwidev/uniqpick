"""Similarity construction: tag TF-IDF and image-feature distances."""

from __future__ import annotations

from typing import Iterable, Mapping

import numpy as np


def cosine_similarity(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    unit = matrix / norms
    return np.clip(unit @ unit.T, -1.0, 1.0)


def build_tfidf(
    docs: Iterable[Mapping[str, float]],
    min_df: int = 1,
    max_df: float = 0.95,
) -> tuple[np.ndarray, list[str], dict[str, int]]:
    """Weighted tag matrix with smoothed IDF and constant tags removed.

    ``max_df`` drops tags that appear in more than that fraction of the corpus
    (artist tags, ratings that slipped through, the style word itself).  Those
    tags cannot discriminate between images and would only inflate every
    similarity.
    """
    docs = list(docs)
    n = len(docs)
    freq: dict[str, int] = {}
    for doc in docs:
        for tag in doc:
            freq[tag] = freq.get(tag, 0) + 1

    ceiling = max_df * n if max_df < 1.0 else n
    vocab = sorted(
        tag for tag, count in freq.items() if count >= min_df and count <= ceiling
    )
    index = {tag: i for i, tag in enumerate(vocab)}
    idf = np.array(
        [np.log((n + 1) / (freq[tag] + 1)) + 1.0 for tag in vocab], dtype=np.float32
    )

    matrix = np.zeros((n, len(vocab)), dtype=np.float32)
    for row, doc in enumerate(docs):
        for tag, weight in doc.items():
            column = index.get(tag)
            if column is not None:
                matrix[row, column] = weight * idf[column]
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms, vocab, freq


def scale01(distances: np.ndarray) -> np.ndarray:
    """Map a symmetric distance matrix into [0, 1] by its observed extremes.

    0 is the closest pair in the corpus and 1 the furthest apart.  Percentile
    clipping is deliberately avoided: with a few thousand pairs the bottom 1%
    all collapses onto 0, which makes unrelated images look identical in the
    report and gives greedy selection hundreds of fake ties.
    """
    if distances.size == 0:
        return distances
    values = distances[np.triu_indices_from(distances, k=1)]
    if values.size == 0:
        return np.zeros_like(distances)
    low = float(values.min())
    high = float(values.max())
    if high - low < 1e-9:
        return np.zeros_like(distances)
    return np.clip((distances - low) / (high - low), 0.0, 1.0)


def tag_distance(docs: Iterable[Mapping[str, float]], min_df: int, max_df: float) -> tuple[np.ndarray, list[str]]:
    matrix, vocab, _ = build_tfidf(docs, min_df=min_df, max_df=max_df)
    if matrix.shape[1] == 0:
        n = matrix.shape[0]
        return np.ones((n, n), dtype=np.float32) - np.eye(n, dtype=np.float32), vocab
    return scale01(1.0 - cosine_similarity(matrix)), vocab


def image_distance(
    embeddings: np.ndarray,
    standardize: bool = False,
) -> np.ndarray:
    """Distance from centred embeddings so shared style does not mask content.

    Every image in a style dataset is close to every other in raw space.  Subtracting
    the corpus mean removes the common direction and leaves what actually differs.
    """
    centred = embeddings - embeddings.mean(axis=0, keepdims=True)
    if standardize:
        spread = centred.std(axis=0, keepdims=True)
        centred = centred / (spread + 1e-6)
    similarity = cosine_similarity(centred)
    return scale01(1.0 - similarity)


# Frame shape, not pixel count, is what survives training: every image is
# resampled into an aspect bucket before the model sees it, so a small and a
# large portrait of the same shape are the same training example.  A corpus
# that is 90% portrait therefore teaches portrait composition along with the
# style, which is why the picker can weight this signal at all.
ASPECT_BUCKETS: tuple[tuple[float, str], ...] = (
    (0.87, "tall"),
    (1.15, "square"),
    (float("inf"), "wide"),
)


def aspect_bucket(ratio: float) -> str:
    """Coarse frame shape: tall below 0.87, square up to 1.15, wide above."""
    for ceiling, name in ASPECT_BUCKETS:
        if ratio < ceiling:
            return name
    return "wide"


def aspect_distance(ratios: np.ndarray) -> np.ndarray:
    """Distance from the log aspect-ratio difference: one doubling counts as 1.

    2:3 versus 3:4 is a small step and 2:3 versus 4:3 saturates at 1.  The
    scale is fixed rather than fitted to the corpus so the number means the
    same thing from one dataset to the next, and the weight in ``combine``
    stays interpretable as "share of the distance that comes from framing".
    """
    logs = np.log2(np.asarray(ratios, dtype=np.float64))
    distance = np.abs(logs[:, None] - logs[None, :])
    return np.clip(distance, 0.0, 1.0).astype(np.float32)


def combine(
    tag_dist: np.ndarray | None,
    image_dist: np.ndarray | None,
    tag_weight: float,
    image_weight: float,
    aspect_dist: np.ndarray | None = None,
    aspect_weight: float = 0.0,
) -> tuple[np.ndarray, float, float, float]:
    """Weighted sum of distances, normalized once at the very end.

    Every signal spans 0..1, so the raw sum spans 0..(tag + image + aspect
    weight).  Picks and hover readouts need a 0..1 number, so the sum is
    divided by the total applied weight as the last step.  Nothing is
    normalized in the middle: tag 1.0 and image 1.0 stay equal partners and
    aspect 0.3 is a soft addition on top, not a share carved out of them.
    A uniform scale cannot move the pick order; the division only keeps the
    reported distances comparable with the per-signal ones.
    """
    parts: list[tuple[np.ndarray, float]] = []
    if tag_dist is not None and tag_weight > 0:
        parts.append((tag_dist, float(tag_weight)))
    if image_dist is not None and image_weight > 0:
        parts.append((image_dist, float(image_weight)))
    if not parts:
        raise ValueError("at least one of tag or image distance is required")

    applied_tag = float(tag_weight) if (tag_dist is not None and tag_weight > 0) else 0.0
    applied_image = float(image_weight) if (image_dist is not None and image_weight > 0) else 0.0
    applied_aspect = float(aspect_weight) if aspect_dist is not None and aspect_weight > 0 else 0.0

    raw = sum(dist * weight for dist, weight in parts)
    if applied_aspect > 0:
        raw = raw + applied_aspect * aspect_dist
    total = applied_tag + applied_image + applied_aspect
    combined = raw / total
    return np.asarray(combined, dtype=np.float32), applied_tag, applied_image, applied_aspect


def uniqueness_scores(distances: np.ndarray) -> np.ndarray:
    """Mean distance to every other image: a cheap standalone 'how unusual' score."""
    n = distances.shape[0]
    if n <= 1:
        return np.zeros(n, dtype=np.float32)
    totals = distances.sum(axis=1) - np.diag(distances)
    return totals / (n - 1)
