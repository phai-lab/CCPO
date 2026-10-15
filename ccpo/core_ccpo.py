"""CCPO step-level advantages from historical and future contextual credit.

Notation follows the paper. For step t of trajectory i on task x_i:

    B_{i,t}[r]  contextual readout of target r over peers P_{i,t}
    C^H_{i,t} = Y_{i,t} - B_{i,t}[Y]                      historical credit
    V_{i,s}   = B_{i,s}[Z] for s < T_i, R_i for s = T_i,  Z_{i,s} = gamma^(T_i-s) R_i
    C^F_{i,t} = z_x[V_{i,t+} - V_{i,t}],  t+ = min(t + K_F, T_i)   future credit
    A_{i,t}   = N_x[C^H + C^F]_{i,t} + alpha_EP * A^EP_{i,t}

All estimation runs in float64 NumPy on CPU.
"""
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, fields
from typing import Literal

import numpy as np

from .context import context_features, trajectory_statistics

EPS = 1e-6


@dataclass(frozen=True)
class CCPOConfig:
    """CCPO hyperparameters. Defaults are the values used in the paper.

    benchmark: normalization convention. "alfworld" z-scores C^H + C^F and A^EP
        within each task. "webshop" leaves C^H + C^F unscaled and mean-centers A^EP.
    episode_weight: alpha_EP. 0 gives CCPO, 1 gives the A_CC + A_EP variant.
    future_horizon: K_F, the number of realized steps spanned by future credit.
    gamma: discount factor of the outcome target Z.
    bandwidth_scale: tau_B = bandwidth_scale * median pairwise distance in B.
    num_removed_pcs: leading principal directions removed from hidden states.
    """
    benchmark: Literal["alfworld", "webshop"] = "alfworld"
    episode_weight: float = 0.0
    future_horizon: int = 2
    gamma: float = 0.95
    bandwidth_scale: float = 0.15
    num_removed_pcs: int = 3

    def __post_init__(self):
        if self.benchmark not in ("alfworld", "webshop"):
            raise ValueError("benchmark must be 'alfworld' or 'webshop'")
        for name, minimum in (("future_horizon", 1), ("num_removed_pcs", 0)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if isinstance(self.episode_weight, bool) or not np.isfinite(self.episode_weight) or self.episode_weight < 0:
            raise ValueError("episode_weight must be a nonnegative number")
        if not np.isfinite(self.gamma) or not 0 <= self.gamma <= 1:
            raise ValueError("gamma must lie in [0, 1]")
        if not np.isfinite(self.bandwidth_scale) or self.bandwidth_scale <= 0:
            raise ValueError("bandwidth_scale must be positive and finite")


@dataclass(frozen=True)
class RolloutBatch:
    """Complete trajectories with one row per step, in any row order.

    task_ids, trajectory_ids, observations: [N] strings x_i, i and o_{i,t}.
        Trajectory IDs must be unique across tasks.
    steps, lengths: [N] integers t and T_i, with 0 <= t < T_i.
    hidden: [N, D] final-layer hidden state h_{i,t} of the frozen reference
        model at the last prompt token.
    returns: [N] historical target Y_{i,t}, the discounted return minus the
        invalid-action penalty of step t.
    outcomes: [N] terminal outcome R_i, repeated on every step of trajectory i.
    episode_scores: [N] episode score R_i minus the invalid-action penalty of
        step t, used only by A^EP.
    Identical copies of a step, such as padding rows, are counted once.
    """
    task_ids: Sequence[str]
    trajectory_ids: Sequence[str]
    observations: Sequence[str]
    steps: np.ndarray
    lengths: np.ndarray
    hidden: np.ndarray
    returns: np.ndarray
    outcomes: np.ndarray
    episode_scores: np.ndarray


@dataclass(frozen=True)
class CCPOResult:
    """Per-row outputs in input row order.

    advantages: A = A_CC + episode_weight * A^EP.
    contextual: A_CC = N_x[C^H + C^F].
    historical, future: C^H and C^F, zero where undefined.
    episode: A^EP, returned even when episode_weight is 0.
    baseline, value, endpoint_value: B[Y], V_{i,t} and V_{i,t+}; NaN where undefined.
    supported: the step has historical or future credit.
    used_fallback: peers came from the task-level fallback.
    """
    advantages: np.ndarray
    contextual: np.ndarray
    historical: np.ndarray
    future: np.ndarray
    episode: np.ndarray
    baseline: np.ndarray
    value: np.ndarray
    endpoint_value: np.ndarray
    supported: np.ndarray
    used_fallback: np.ndarray


def _validate(batch: RolloutBatch) -> RolloutBatch:
    n = len(batch.task_ids)
    if n == 0:
        raise ValueError("Rollout batch must not be empty")
    arrays = {}
    for name in ("task_ids", "trajectory_ids", "observations"):
        value = np.asarray(getattr(batch, name))
        if value.shape != (n,) or not all(isinstance(item, (str, bytes)) for item in value):
            raise ValueError(f"{name} must contain N strings")
        arrays[name] = value.astype(str)
    for name in ("steps", "lengths"):
        value = np.asarray(getattr(batch, name))
        if value.shape != (n,) or value.dtype.kind not in "iu":
            raise ValueError(f"{name} must contain N integers")
        arrays[name] = value.astype(np.int64)
    if np.any(arrays["steps"] < 0) or np.any(arrays["steps"] >= arrays["lengths"]):
        raise ValueError("Each step must lie in [0, trajectory length)")
    for name in ("returns", "outcomes", "episode_scores", "hidden"):
        value = np.asarray(getattr(batch, name), dtype=np.float64)
        if name == "hidden":
            valid_shape = value.ndim == 2 and value.shape[0] == n and value.shape[1] > 0
        else:
            valid_shape = value.shape == (n,)
        if not valid_shape or not np.isfinite(value).all():
            raise ValueError(f"{name} has an invalid shape or non-finite values")
        arrays[name] = value
    return RolloutBatch(**arrays)


def _unique_steps(batch: RolloutBatch):
    """Keep one row per (task, trajectory, step) and check trajectories are complete.

    Returns the unique rows sorted by trajectory and step, the index that maps
    them back to input rows, and the unique row indices of each trajectory.
    """
    first, owners, lengths = {}, {}, {}
    keys = []
    for row, (task, trajectory, step, length) in enumerate(zip(
        batch.task_ids, batch.trajectory_ids, batch.steps, batch.lengths,
    )):
        if trajectory in owners and owners[trajectory] != task:
            raise ValueError("Trajectory IDs must be unique across tasks")
        if trajectory in lengths and lengths[trajectory] != length:
            raise ValueError("Inconsistent lengths within a trajectory")
        owners[trajectory], lengths[trajectory] = task, length
        key = (task, trajectory, step)
        first.setdefault(key, row)
        keys.append(key)
    ordered = sorted(first)
    position = {key: row for row, key in enumerate(ordered)}
    take = np.array([first[key] for key in ordered])
    restore = np.array([position[key] for key in keys])
    unique = RolloutBatch(**{f.name: getattr(batch, f.name)[take] for f in fields(batch)})
    for field in fields(batch):
        if not np.array_equal(getattr(batch, field.name), getattr(unique, field.name)[restore]):
            raise ValueError(f"Copies of the same step disagree on {field.name}")
    groups = defaultdict(list)
    for row, trajectory in enumerate(unique.trajectory_ids):
        groups[trajectory].append(row)
    for trajectory, rows in groups.items():
        if unique.steps[rows].tolist() != list(range(lengths[trajectory])):
            raise ValueError(f"Incomplete trajectory: {trajectory}")
        if not np.all(unique.outcomes[rows] == unique.outcomes[rows[0]]):
            raise ValueError("Outcome must be constant within a trajectory")
    return unique, restore, [np.array(rows) for rows in groups.values()]


def _peer_readouts(batch: RolloutBatch, phi: np.ndarray, targets: np.ndarray, bandwidth_scale: float):
    """Contextual readouts B_{i,t}[r] for each column r of targets.

    Peers share the task and observation of the query step and come from other
    trajectories. A step without such peers falls back to all steps of its task
    in other trajectories. The kernel bandwidth uses every pair in the bucket,
    before the query trajectory is excluded. Repeated visits keep individual
    weights: each peer trajectory contributes its weighted mean, weighted again
    by its total kernel mass.

    Returns readouts (NaN without peers) and the bucket level used for each
    row: 0 for exact observation peers, 1 for the task fallback, -1 for none.
    """
    n = len(phi)
    readout = np.full((n, targets.shape[1]), np.nan)
    level = np.full(n, -1, dtype=int)
    bucket_keys = (zip(batch.task_ids, batch.observations), batch.task_ids)
    for depth, keys in enumerate(bucket_keys):
        buckets = defaultdict(list)
        for row, key in enumerate(keys):
            buckets[key].append(row)
        for rows in buckets.values():
            rows = np.array(rows)
            trajectories = batch.trajectory_ids[rows]
            if len(set(trajectories)) < 2 or np.all(level[rows] >= 0):
                continue
            z = phi[rows]
            squared = (z * z).sum(axis=1)
            distances = np.sqrt(np.maximum(squared[:, None] + squared[None, :] - 2 * (z @ z.T), 0))
            median = np.median(distances[np.triu_indices(len(rows), 1)])
            bandwidth = bandwidth_scale * (median if median > 1e-9 else 1.)
            for local, row in enumerate(rows):
                if level[row] >= 0:
                    continue
                peers = np.flatnonzero(trajectories != trajectories[local])
                weights = np.exp(-distances[local, peers] / bandwidth)
                masses, sums = {}, {}
                for weight, peer in zip(weights, peers):
                    trajectory = trajectories[peer]
                    masses[trajectory] = masses.get(trajectory, 0.) + weight
                    sums[trajectory] = sums.get(trajectory, np.zeros(targets.shape[1])) + weight * targets[rows[peer]]
                mass = np.array(list(masses.values()))
                means = np.stack([sums[t] / max(masses[t], 1e-12) for t in masses])
                readout[row] = (mass[:, None] * means).sum(axis=0) / max(mass.sum(), 1e-12)
                level[row] = depth
    return readout, level


def _task_normalize(values, task_ids, active, *, divide_std, keep_singleton):
    """Center (and optionally scale) active rows within each task.

    Uses the sample standard deviation plus EPS. Tasks with fewer than two
    active rows keep their values when keep_singleton is set and are zero
    otherwise; inactive rows follow the same rule.
    """
    result = values.copy() if keep_singleton else np.zeros_like(values)
    for task in np.unique(task_ids):
        rows = np.flatnonzero((task_ids == task) & active)
        if len(rows) < 2:
            continue
        centered = values[rows] - values[rows].mean()
        result[rows] = centered / (values[rows].std(ddof=1) + EPS) if divide_std else centered
    return result


def _episode_advantage(scores, task_ids, *, divide_std):
    """GRPO-style episode advantage A^EP over all rows of each task.

    As in GRPO, a task with a single row uses mean 0 and standard deviation 1.
    """
    episode = _task_normalize(scores, task_ids, np.ones(len(scores), dtype=bool),
                              divide_std=divide_std, keep_singleton=True)
    if divide_std:
        for task in np.unique(task_ids):
            rows = np.flatnonzero(task_ids == task)
            if len(rows) == 1:
                episode[rows] /= 1 + EPS
    return episode


def compute_ccpo_advantage(batch: RolloutBatch, config: CCPOConfig = CCPOConfig()) -> CCPOResult:
    """Compute CCPO advantages for a globally gathered batch of complete trajectories.

    Features, peer readouts and future-credit standardization use unique steps.
    The final contextual and episode normalizations use all input rows, so
    longer trajectories contribute more rows. Outputs follow input row order.
    """
    rows = _validate(batch)
    unique, restore, trajectories = _unique_steps(rows)
    statistics = trajectory_statistics(unique.observations, unique.trajectory_ids, unique.steps)
    phi = context_features(unique.hidden, statistics, config.num_removed_pcs)
    outcome_targets = config.gamma ** (unique.lengths - unique.steps) * unique.outcomes
    readout, level = _peer_readouts(
        unique, phi, np.column_stack([unique.returns, outcome_targets]), config.bandwidth_scale,
    )
    has_peers = level >= 0
    baseline, value = readout[:, 0], readout[:, 1]
    historical = np.where(has_peers, unique.returns - baseline, 0.)

    endpoint_value = unique.outcomes.copy()
    for trajectory in trajectories:
        for step, row in enumerate(trajectory):
            endpoint = step + config.future_horizon
            if endpoint < len(trajectory):
                endpoint_value[row] = value[trajectory[endpoint]]
    eligible = np.isfinite(value) & np.isfinite(endpoint_value)
    delta = np.where(eligible, endpoint_value - value, 0.)
    future = _task_normalize(delta, unique.task_ids, eligible, divide_std=True, keep_singleton=False)

    supported = (has_peers | eligible)[restore]
    historical, future = historical[restore], future[restore]
    contextual = historical + future
    alfworld = config.benchmark == "alfworld"
    if alfworld:
        contextual = _task_normalize(contextual, rows.task_ids, supported, divide_std=True, keep_singleton=True)
    episode = _episode_advantage(rows.episode_scores, rows.task_ids, divide_std=alfworld)
    advantages = contextual + config.episode_weight * episode
    if not np.isfinite(advantages).all():
        raise ValueError("Non-finite CCPO advantages")
    return CCPOResult(
        advantages=advantages, contextual=contextual, historical=historical, future=future,
        episode=episode, baseline=baseline[restore], value=value[restore],
        endpoint_value=endpoint_value[restore], supported=supported,
        used_fallback=(level[restore] == 1),
    )
