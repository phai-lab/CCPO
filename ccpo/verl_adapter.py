"""PyTorch helpers for computing CCPO advantages on verl-agent rollout batches."""
from typing import Any

import numpy as np
import torch

from .core_ccpo import CCPOConfig, CCPOResult, RolloutBatch, compute_ccpo_advantage

HIDDEN_KEY = "ccpo_hidden"
STEP_KEY = "ccpo_step_index"


def last_prompt_hidden(
    hidden: torch.Tensor, attention_mask: torch.Tensor, response_length: int,
    packed_indices: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return detached float32 hidden states h_{i,t} at the last prompt token.

    Padded input: hidden [N,L,D] and attention_mask [N,L], with left-padded
    prompts followed by right-padded responses of width response_length.
    Packed input: hidden [1,nnz(mask),D], with packed_indices equal to the
    flattened nonzero indices of attention_mask. Pass the final-layer hidden
    states of the frozen reference model.
    """
    if attention_mask.ndim != 2 or hidden.ndim != 3:
        raise ValueError("Expected attention_mask [N,L] and hidden [N,L,D] or [1,P,D]")
    n, width = attention_mask.shape
    if isinstance(response_length, bool) or not isinstance(response_length, int) or not 0 < response_length < width:
        raise ValueError("response_length must leave at least one prompt token")
    column = width - response_length - 1
    if not bool(attention_mask[:, column].bool().all()):
        raise ValueError("Last prompt token is masked")
    if packed_indices is None:
        if hidden.shape[:2] != (n, width):
            raise ValueError("Padded hidden states do not match inputs")
        result = hidden[:, column]
    else:
        expected = attention_mask.reshape(-1).nonzero().flatten()
        if not torch.equal(packed_indices, expected) or hidden.shape[:2] != (1, expected.numel()):
            raise ValueError("Packed hidden states or indices do not match inputs")
        positions = torch.searchsorted(expected, torch.arange(n, device=expected.device) * width + column)
        result = hidden[0, positions]
    if result.shape[-1] == 0 or not bool(torch.isfinite(result).all()):
        raise ValueError("Hidden states are empty or non-finite")
    return result.detach().float().cpu()


def token_advantages(result: CCPOResult, response_mask: torch.Tensor) -> torch.Tensor:
    """Broadcast each step advantage to all of its response tokens.

    This covers reasoning and action tokens. Prompt tokens lie outside the
    response mask and padding tokens receive zero. Returns detached float32
    on response_mask.device.
    """
    if response_mask.ndim != 2 or response_mask.shape[0] != len(result.advantages):
        raise ValueError("response_mask must have shape [number of steps, response length]")
    if not bool(((response_mask == 0) | (response_mask == 1)).all()) or not bool((response_mask.sum(-1) > 0).all()):
        raise ValueError("Each response must have a nonempty binary token mask")
    values = torch.as_tensor(result.advantages, device=response_mask.device, dtype=torch.float32)
    return (values[:, None] * response_mask.detach()).detach()


def compute_verl_advantages(
    data: Any, config: CCPOConfig = CCPOConfig(),
) -> tuple[torch.Tensor, CCPOResult]:
    """Return (token advantages, CCPOResult) for a verl-agent DataProto.

    The batch must hold complete, globally gathered trajectories.

    data.batch:
        response_mask [N,L], token_level_rewards [N,L] whose row sums are the
        episode scores, step_rewards [N] or [N,L] holding Y_{i,t}, and
        ccpo_hidden [N,D] from last_prompt_hidden.
    data.non_tensor_batch:
        uid (task), traj_uid, anchor_obs, ccpo_step_index, episode_lengths and
        episode_rewards (R_i).
    """
    tensors, metadata = data.batch, data.non_tensor_batch
    mask = tensors["response_mask"]
    rewards = tensors["token_level_rewards"]
    if mask.ndim != 2 or rewards.shape != mask.shape:
        raise ValueError("token_level_rewards must match response_mask [N,L]")
    if bool((rewards[mask == 0] != 0).any()):
        raise ValueError("Masked response positions must not contain rewards")
    returns = tensors["step_rewards"]
    if returns.ndim == 2:
        if returns.shape != mask.shape:
            raise ValueError("Token-shaped step_rewards must match response_mask")
        returns = (returns * mask).sum(-1)

    def numpy(value):
        return value.detach().float().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)

    def integers(name):
        # verl-agent stores episode lengths as integral floats in object arrays.
        raw = np.asarray(metadata[name])
        values = np.asarray(raw, dtype=np.float64)
        if (raw.ndim != 1 or any(isinstance(x, (bool, np.bool_)) for x in raw)
                or not np.isfinite(values).all() or np.any(values != np.floor(values))
                or np.any(values < 0) or np.any(values >= 2**63)):
            raise ValueError(f"{name} must contain nonnegative integral metadata")
        return values.astype(np.int64)

    batch = RolloutBatch(
        task_ids=metadata["uid"], trajectory_ids=metadata["traj_uid"],
        observations=metadata["anchor_obs"], steps=integers(STEP_KEY),
        lengths=integers("episode_lengths"), hidden=numpy(tensors[HIDDEN_KEY]),
        returns=numpy(returns), outcomes=numpy(metadata["episode_rewards"]),
        episode_scores=numpy(rewards.sum(-1)),
    )
    result = compute_ccpo_advantage(batch, config)
    return token_advantages(result, mask), result
