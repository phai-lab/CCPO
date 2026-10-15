"""Context representation phi_{i,t} used to weight observation-matched peers."""
from collections import defaultdict
from collections.abc import Sequence

import numpy as np


def trajectory_statistics(
    observations: Sequence[str], trajectory_ids: Sequence[str], steps: np.ndarray,
) -> np.ndarray:
    """Return [t, n_{i,t}, revisit_{i,t}, n_{i,t}/(t+1)] for every row.

    n_{i,t} is the number of distinct observations in o_{i,0..t}, and revisit
    marks an observation that appeared earlier in the same trajectory.
    Observations are compared by exact string equality.
    """
    if len(observations) != len(trajectory_ids) or len(steps) != len(observations):
        raise ValueError("Context fields must have the same number of rows")
    groups = defaultdict(list)
    for row, trajectory in enumerate(trajectory_ids):
        groups[trajectory].append(row)
    result = np.zeros((len(steps), 4), dtype=np.float64)
    for rows in groups.values():
        seen = set()
        for row in sorted(rows, key=lambda i: steps[i]):
            observation = observations[row]
            revisit = observation in seen
            seen.add(observation)
            result[row] = (steps[row], len(seen), revisit, len(seen) / (steps[row] + 1))
    return result


def _unit_rows(values: np.ndarray) -> np.ndarray:
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-9)


def _thermometer(values: np.ndarray, upper: float) -> np.ndarray:
    count = (12 * np.clip(values / upper, 0, 1)).astype(int)
    return (np.arange(12)[None, :] < count[:, None]).astype(np.float64)


def context_features(
    hidden: np.ndarray, statistics: np.ndarray, num_removed_pcs: int = 3,
) -> np.ndarray:
    """Return phi = nu([nu(P(h - mean(h))); nu(c)]) for every row.

    Hidden states h are centered over the batch, their leading principal
    directions are removed, and the result is L2-normalized. The summary c has
    37 coordinates: thermometer encodings of t, n_{i,t} and n_{i,t}/(t+1) with
    12 coordinates each and upper bounds 30, 25 and 1, plus the revisit bit.
    Both blocks have equal weight. No reward enters this construction.
    With at most num_removed_pcs rows no directions are removed, and a single
    row keeps its raw hidden state.
    """
    hidden = np.asarray(hidden, dtype=np.float64)
    statistics = np.asarray(statistics, dtype=np.float64)
    if hidden.ndim != 2 or not all(hidden.shape) or statistics.shape != (len(hidden), 4):
        raise ValueError("Expected hidden [N,D] and statistics [N,4], with N,D > 0")
    if not np.isfinite(hidden).all() or not np.isfinite(statistics).all():
        raise ValueError("Features must be finite")
    if (isinstance(num_removed_pcs, bool) or not isinstance(num_removed_pcs, (int, np.integer))
            or num_removed_pcs < 0):
        raise ValueError("num_removed_pcs must be a nonnegative integer")
    encoded = hidden.copy()
    if len(encoded) >= 2:
        encoded -= encoded.mean(axis=0, keepdims=True)
        if num_removed_pcs and len(encoded) > num_removed_pcs:
            _, _, directions = np.linalg.svd(encoded, full_matrices=False)
            principal = directions[:num_removed_pcs]
            encoded -= encoded @ principal.T @ principal
        encoded = _unit_rows(encoded)
    summary = np.concatenate([
        _thermometer(statistics[:, 0], 30),
        _thermometer(statistics[:, 1], 25),
        _thermometer(statistics[:, 3], 1),
        (statistics[:, 2:3] > 0).astype(np.float64),
    ], axis=1)
    return _unit_rows(np.concatenate([encoded, _unit_rows(summary)], axis=1))
