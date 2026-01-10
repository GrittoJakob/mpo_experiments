import numpy
import torch

def sample_minibatch(replaybuffer, batch_size: int, device: torch.device, gpu_buffer: bool):
    
    s, a, ns, r = replaybuffer.sample_batch_stacked(batch_size, replace=True)

    if gpu_buffer or isinstance(s, torch.Tensor):
        
        state_batch      = s.to(device=device, dtype=torch.float32, non_blocking=True)
        action_batch     = a.to(device=device, dtype=torch.float32, non_blocking=True)
        next_state_batch = ns.to(device=device, dtype=torch.float32, non_blocking=True)
        reward_batch     = r.to(device=device, dtype=torch.float32, non_blocking=True)
    else:
        # numpy -> torch
        state_batch      = torch.from_numpy(s).to(device=device, dtype=torch.float32, non_blocking=True)
        action_batch     = torch.from_numpy(a).to(device=device, dtype=torch.float32, non_blocking=True)
        next_state_batch = torch.from_numpy(ns).to(device=device, dtype=torch.float32, non_blocking=True)
        reward_batch     = torch.from_numpy(r).to(device=device, dtype=torch.float32, non_blocking=True)

    return state_batch, action_batch, next_state_batch, reward_batch

def assert_batch_shapes(state_batch, action_batch, next_state_batch, reward_batch, batch_size, state_dim, action_dim):
    assert state_batch.ndim == 2, f"state_batch.ndim={state_batch.ndim}, shape={tuple(state_batch.shape)}"
    assert next_state_batch.ndim == 2, f"next_state_batch.ndim={next_state_batch.ndim}, shape={tuple(next_state_batch.shape)}"
    assert action_batch.ndim == 2, f"action_batch.ndim={action_batch.ndim}, shape={tuple(action_batch.shape)}"

    assert state_batch.shape == (batch_size, state_dim), f"state_batch shape {tuple(state_batch.shape)} != {(batch_size, state_dim)}"
    assert next_state_batch.shape == (batch_size, state_dim), f"next_state_batch shape {tuple(next_state_batch.shape)} != {(batch_size, state_dim)}"
    assert action_batch.shape == (batch_size, action_dim), f"action_batch shape {tuple(action_batch.shape)} != {(batch_size, action_dim)}"

    # reward: allow (B,) or (B,1) but normalize to (B,)
    assert reward_batch.ndim in (1, 2), f"reward_batch.ndim={reward_batch.ndim}, shape={tuple(reward_batch.shape)}"
    if reward_batch.ndim == 2:
        assert reward_batch.shape[1] == 1, f"reward_batch shape {tuple(reward_batch.shape)} expected (B,1)"
