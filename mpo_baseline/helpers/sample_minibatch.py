import numpy
import torch

def sample_minibatch(replaybuffer, batch_size: int, device: torch.device, gpu_buffer: bool):
    """
    Samples a minibatch of transitions.

    Expected replaybuffer output (stacked):
      s, a, ns, r, terminated, truncated

    Returns tensors on `device`:
      state_batch:      (B, obs_dim)
      action_batch:     (B, act_dim)
      next_state_batch: (B, obs_dim)
      reward_batch:     (B, 1)
      terminated_batch: (B, 1)  float {0,1}
      truncated_batch:  (B, 1)  float {0,1}
    """

    s, a, ns, r, terminated, truncated = replaybuffer.sample_batch_stacked(batch_size, replace=True)

    def to_torch(x):
        if isinstance(x, torch.Tensor):
            t = x
        else:
            # numpy -> torch
            t = torch.from_numpy(x)
        return t.to(device=device, dtype=torch.float32, non_blocking=True)

    state_batch      = to_torch(s)
    action_batch     = to_torch(a)
    next_state_batch = to_torch(ns)
    reward_batch     = to_torch(r)
    terminated_batch = to_torch(terminated)
    truncated_batch  = to_torch(truncated)

    # Normalize shapes to (B,1) for reward and flags
    if reward_batch.ndim == 1:
        reward_batch = reward_batch.view(-1, 1)
    elif reward_batch.ndim == 2 and reward_batch.shape[1] != 1:
        reward_batch = reward_batch.reshape(-1, 1)

    if terminated_batch.ndim == 1:
        terminated_batch = terminated_batch.view(-1, 1)
    elif terminated_batch.ndim == 2 and terminated_batch.shape[1] != 1:
        terminated_batch = terminated_batch.reshape(-1, 1)

    if truncated_batch.ndim == 1:
        truncated_batch = truncated_batch.view(-1, 1)
    elif truncated_batch.ndim == 2 and truncated_batch.shape[1] != 1:
        truncated_batch = truncated_batch.reshape(-1, 1)

    return state_batch, action_batch, next_state_batch, reward_batch, terminated_batch, truncated_batch


def assert_batch_shapes(
    state_batch,
    action_batch,
    next_state_batch,
    reward_batch,
    terminated_batch,
    truncated_batch,
    batch_size,
    state_dim,
    action_dim,
):
    assert state_batch.ndim == 2, f"state_batch.ndim={state_batch.ndim}, shape={tuple(state_batch.shape)}"
    assert next_state_batch.ndim == 2, f"next_state_batch.ndim={next_state_batch.ndim}, shape={tuple(next_state_batch.shape)}"
    assert action_batch.ndim == 2, f"action_batch.ndim={action_batch.ndim}, shape={tuple(action_batch.shape)}"

    assert terminated_batch.min() >= 0 and terminated_batch.max() <= 1
    assert truncated_batch.min() >= 0 and truncated_batch.max() <= 1


    assert state_batch.shape == (batch_size, state_dim), \
        f"state_batch shape {tuple(state_batch.shape)} != {(batch_size, state_dim)}"
    assert next_state_batch.shape == (batch_size, state_dim), \
        f"next_state_batch shape {tuple(next_state_batch.shape)} != {(batch_size, state_dim)}"
    assert action_batch.shape == (batch_size, action_dim), \
        f"action_batch shape {tuple(action_batch.shape)} != {(batch_size, action_dim)}"

    # reward and flags must be (B,1)
    assert reward_batch.ndim == 2 and reward_batch.shape == (batch_size, 1), \
        f"reward_batch shape {tuple(reward_batch.shape)} expected {(batch_size, 1)}"
    assert terminated_batch.ndim == 2 and terminated_batch.shape == (batch_size, 1), \
        f"terminated_batch shape {tuple(terminated_batch.shape)} expected {(batch_size, 1)}"
    assert truncated_batch.ndim == 2 and truncated_batch.shape == (batch_size, 1), \
        f"truncated_batch shape {tuple(truncated_batch.shape)} expected {(batch_size, 1)}"
