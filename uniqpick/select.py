"""Selection strategies: greedy coverage, max-min (k-center) and disparity.

Every method starts from a greedy walk and, unless ``refine`` is off, then
improves the set by swapping picks with unpicked images for as long as the
objective rises.  Greedy alone is a (1 - 1/e) approximation for coverage and
can leave an obviously better set on the table; the swap pass is a local
search over the final set, so the picks are always at least as good as the
raw greedy order.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

METHODS = ("coverage", "kcenter", "disparity")


@dataclass
class Selection:
    """Ordered picks plus the score that earned each slot."""

    order: list[int]
    scores: list[float]
    nearest: list[float]
    method: str
    objective: float = 0.0
    refined: bool = False


def _clean(scores: np.ndarray, chosen: list[int]) -> np.ndarray:
    scores = scores.copy()
    scores[chosen] = -np.inf
    return scores


def _outside(n: int, chosen: list[int]) -> np.ndarray:
    mask = np.ones(n, dtype=bool)
    mask[chosen] = False
    return np.flatnonzero(mask)


def coverage_objective(similarity: np.ndarray, chosen: list[int]) -> float:
    """Sum of each corpus image's best similarity to the set."""
    return float(similarity[:, chosen].max(axis=1).sum())


def kcenter_objective(distance: np.ndarray, chosen: list[int]) -> float:
    """Distance of the closest pair inside the set (the weakest link)."""
    if len(chosen) < 2:
        return 0.0
    sub = distance[np.ix_(chosen, chosen)]
    upper = np.triu_indices(len(chosen), 1)
    return float(sub[upper].min())


def disparity_objective(distance: np.ndarray, chosen: list[int]) -> float:
    """Sum of all pairwise distances inside the set."""
    if len(chosen) < 2:
        return 0.0
    sub = distance[np.ix_(chosen, chosen)]
    upper = np.triu_indices(len(chosen), 1)
    return float(sub[upper].sum())


def _refine_coverage(similarity: np.ndarray, chosen: list[int], rounds: int = 64) -> list[int]:
    """Swap one pick for one unpicked image while that improves coverage."""
    chosen = list(chosen)
    n = similarity.shape[0]
    if not 2 <= len(chosen) < n:
        return chosen

    tol = 1e-9
    for _ in range(rounds):
        sub = similarity[:, chosen]
        best = sub.max(axis=1)
        owner = sub.argmax(axis=1)
        second = np.partition(sub, -2, axis=1)[:, -2]
        outside = _outside(n, chosen)

        best_gain, best_pos, best_add = tol, -1, -1
        for pos in range(len(chosen)):
            # Removing this pick only hurts the images for which it was the
            # nearest one; they fall back to their second best.
            base = np.where(owner == pos, second, best)
            gains = np.maximum(similarity[:, outside] - base[:, None], 0.0).sum(axis=0)
            add = int(np.argmax(gains))
            if gains[add] > best_gain:
                best_gain, best_pos, best_add = float(gains[add]), pos, int(outside[add])

        if best_pos < 0:
            break
        chosen[best_pos] = best_add

    return chosen


def _refine_kcenter(distance: np.ndarray, chosen: list[int], rounds: int = 64) -> list[int]:
    """Raise the smallest distance between two picks by swapping one of them out."""
    chosen = list(chosen)
    n = distance.shape[0]
    if not 2 <= len(chosen) < n:
        return chosen

    tol = 1e-9
    for _ in range(rounds):
        current = kcenter_objective(distance, chosen)
        outside = _outside(n, chosen)

        best_gain, best_pos, best_add = tol, -1, -1
        for pos in range(len(chosen)):
            rest = np.array([c for q, c in enumerate(chosen) if q != pos])
            if len(rest) < 2:
                weakest = np.inf
            else:
                within = distance[np.ix_(rest, rest)]
                weakest = float(within[np.triu_indices(len(rest), 1)].min())
            rows = distance[np.ix_(outside, rest)].min(axis=1)
            values = np.minimum(weakest, rows) - current
            add = int(np.argmax(values))
            if values[add] > best_gain:
                best_gain, best_pos, best_add = float(values[add]), pos, int(outside[add])

        if best_pos < 0:
            break
        chosen[best_pos] = best_add

    return chosen


def _refine_disparity(distance: np.ndarray, chosen: list[int], rounds: int = 64) -> list[int]:
    """Swap out a pick whose distances to the rest sum to little."""
    chosen = list(chosen)
    n = distance.shape[0]
    if not 2 <= len(chosen) < n:
        return chosen

    tol = 1e-9
    for _ in range(rounds):
        outside = _outside(n, chosen)
        best_gain, best_pos, best_add = tol, -1, -1
        for pos in range(len(chosen)):
            rest = [c for q, c in enumerate(chosen) if q != pos]
            sums = distance[:, rest].sum(axis=1)
            values = sums[outside] - sums[chosen[pos]]
            add = int(np.argmax(values))
            if values[add] > best_gain:
                best_gain, best_pos, best_add = float(values[add]), pos, int(outside[add])

        if best_pos < 0:
            break
        chosen[best_pos] = best_add

    return chosen


def _replay_coverage(similarity: np.ndarray, order: list[int]) -> tuple[list[float], list[float]]:
    """Rewrite the per-slot scores for the final order.

    ``scores`` stay what they always meant: the coverage this pick added when
    it joined the set, so a refined set still explains every slot.
    """
    scores: list[float] = []
    nearest: list[float] = []
    total = 0.0
    for pos, pick in enumerate(order):
        if pos == 0:
            scores.append(coverage_objective(similarity, [pick]))
            nearest.append(0.0)
            total = scores[0]
            continue
        new_total = float(similarity[:, order[: pos + 1]].max(axis=1).sum())
        scores.append(new_total - total)
        nearest.append(float(1.0 - similarity[pick, order[:pos]].max()))
        total = new_total
    return scores, nearest


def _replay_kcenter(distance: np.ndarray, order: list[int]) -> tuple[list[float], list[float]]:
    scores: list[float] = []
    nearest: list[float] = []
    for pos, pick in enumerate(order):
        if pos == 0:
            scores.append(0.0)
            nearest.append(0.0)
            continue
        value = float(distance[pick, order[:pos]].min())
        scores.append(value)
        nearest.append(value)
    return scores, nearest


def _replay_disparity(distance: np.ndarray, order: list[int]) -> tuple[list[float], list[float]]:
    scores: list[float] = []
    nearest: list[float] = []
    for pos, pick in enumerate(order):
        if pos == 0:
            scores.append(0.0)
            nearest.append(0.0)
            continue
        scores.append(float(distance[pick, order[:pos]].sum()))
        nearest.append(float(distance[pick, order[:pos]].min()))
    return scores, nearest


def select_coverage(similarity: np.ndarray, k: int, refine: bool = True) -> Selection:
    """Greedy facility location: maximize how well the picks cover the corpus.

    Each new pick earns the summed similarity it adds for every image not yet
    well covered.  This favours a representative spread over a set of oddities,
    which is what keeps one feature from dominating the training set.
    """
    n = similarity.shape[0]
    k = max(1, min(k, n))
    first = int(np.argmax(similarity.sum(axis=1)))
    chosen = [first]
    covered = similarity[first].copy()

    while len(chosen) < k:
        gain = np.maximum(similarity - covered[None, :], 0.0).sum(axis=1)
        gain = _clean(gain, chosen)
        nxt = int(np.argmax(gain))
        chosen.append(nxt)
        covered = np.maximum(covered, similarity[nxt])

    refined = False
    if refine:
        swapped = _refine_coverage(similarity, chosen)
        refined = swapped != chosen
        chosen = swapped

    scores, nearest = _replay_coverage(similarity, chosen)
    return Selection(chosen, scores, nearest, "coverage",
                     objective=coverage_objective(similarity, chosen), refined=refined)


def select_kcenter(distance: np.ndarray, k: int, refine: bool = True) -> Selection:
    """Greedy max-min (farthest point sampling): each pick is the image farthest
    from everything chosen so far.  This maximizes spread but tends to surface
    outliers and rare crops rather than typical examples.
    """
    n = distance.shape[0]
    k = max(1, min(k, n))
    first = int(np.argmin(distance.sum(axis=1)))
    chosen = [first]
    nearest_to_set = distance[first].copy()

    while len(chosen) < k:
        candidate = _clean(nearest_to_set, chosen)
        nxt = int(np.argmax(candidate))
        chosen.append(nxt)
        nearest_to_set = np.minimum(nearest_to_set, distance[nxt])

    refined = False
    if refine:
        swapped = _refine_kcenter(distance, chosen)
        refined = swapped != chosen
        chosen = swapped

    scores, nearest = _replay_kcenter(distance, chosen)
    return Selection(chosen, scores, nearest, "kcenter",
                     objective=kcenter_objective(distance, chosen), refined=refined)


def select_disparity(distance: np.ndarray, k: int, refine: bool = True) -> Selection:
    """Sum-of-pairs: maximize the total distance inside the set.

    Greedy adds the image whose distances to the current picks add up to the
    most.  This spreads picks more evenly than max-min (which only cares about
    the single weakest link) but still prefers edges of the corpus, so it is
    best for "show me the extremes" rather than "show me the middle".
    """
    n = distance.shape[0]
    k = max(1, min(k, n))
    first = int(np.argmin(distance.sum(axis=1)))
    chosen = [first]
    total = distance[first].copy()

    while len(chosen) < k:
        candidate = _clean(total, chosen)
        nxt = int(np.argmax(candidate))
        chosen.append(nxt)
        total = total + distance[nxt]

    refined = False
    if refine:
        swapped = _refine_disparity(distance, chosen)
        refined = swapped != chosen
        chosen = swapped

    scores, nearest = _replay_disparity(distance, chosen)
    return Selection(chosen, scores, nearest, "disparity",
                     objective=disparity_objective(distance, chosen), refined=refined)


def select(distance: np.ndarray, k: int, method: str = "coverage", refine: bool = True) -> Selection:
    if method == "coverage":
        return select_coverage(1.0 - distance, k, refine=refine)
    if method == "kcenter":
        return select_kcenter(distance, k, refine=refine)
    if method == "disparity":
        return select_disparity(distance, k, refine=refine)
    raise ValueError(f"unknown method: {method}")


def nearest_selected(distance: np.ndarray, chosen: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """For every image, the closest chosen image and that distance."""
    sub = distance[:, chosen]
    positions = np.argmin(sub, axis=1)
    rows = np.arange(distance.shape[0])
    return np.array(chosen)[positions], sub[rows, positions]
