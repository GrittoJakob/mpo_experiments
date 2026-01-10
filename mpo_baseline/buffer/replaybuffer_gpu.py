import torch
from typing import Any, Iterable, List, Optional, Sequence, Tuple
import numpy as np


class ReplayBufferGPU:
    """
    Episodic replay buffer on GPU.

    - Stores complete episodes (only after done_episode/store_episodes), like your CPU version.
    - Keeps a flat transition ring-buffer on GPU (preallocated) for fast sampling.
    - Evicts whole episodes from the front to stay under max_transitions.

    API mirrors the CPU buffer:
      store_step, done_episode, store_episodes
      sample_batch, sample_batch_stacked, batch_get
      __len__, __getitem__
      mean_reward, mean_return
      + mean_episode_length (new)
    """

    def __init__(
        self,
        max_replay_buffer: int = 1_000_000,
        device: str = "cuda",
        dtype: torch.dtype = torch.float32,
    ):
        self.max_transitions = int(max_replay_buffer)
        self.device = torch.device(device)
        self.dtype = dtype

        # Lazy init shapes on first episode
        self._state_dim: Optional[int] = None
        self._action_dim: Optional[int] = None

        # Flat ring buffer storage (allocated lazily)
        self._states: Optional[torch.Tensor] = None
        self._actions: Optional[torch.Tensor] = None
        self._next_states: Optional[torch.Tensor] = None
        self._rewards: Optional[torch.Tensor] = None  # (N, 1)

        # Ring pointers (logical buffer has [start .. start+size) mod N)
        self._start: int = 0
        self._size: int = 0

        # Episode metadata (FIFO order of stored episodes)
        # Each entry: (usable_len, return_sum, mean_reward)
        self._episodes_meta: List[Tuple[int, float, float]] = []

        # Temporary episode storage (CPU objects or tensors) until done_episode
        self.tmp_episode_buff: List[Tuple[Any, Any, Any, Any]] = []

    # ---------- Internal helpers ----------

    def _ensure_allocated(self, state_example, action_example):
        if self._states is not None:
            return

        # Infer dims
        s = torch.as_tensor(state_example)
        a = torch.as_tensor(action_example)

        if s.ndim != 1:
            s = s.reshape(-1)
        if a.ndim != 1:
            a = a.reshape(-1)

        self._state_dim = int(s.numel())
        self._action_dim = int(a.numel())

        N = self.max_transitions
        sd, ad = self._state_dim, self._action_dim

        self._states = torch.empty((N, sd), device=self.device, dtype=self.dtype)
        self._actions = torch.empty((N, ad), device=self.device, dtype=self.dtype)
        self._next_states = torch.empty((N, sd), device=self.device, dtype=self.dtype)
        self._rewards = torch.empty((N, 1), device=self.device, dtype=self.dtype)

    def _write_block(self, buf: torch.Tensor, data: torch.Tensor, write_pos: int):
        """
        Write `data` (T, D) into ring buffer `buf` starting at write_pos (mod N).
        """
        N = buf.shape[0]
        T = data.shape[0]
        if T <= 0:
            return

        end_space = N - write_pos
        if T <= end_space:
            buf[write_pos : write_pos + T] = data
        else:
            buf[write_pos:] = data[:end_space]
            buf[: T - end_space] = data[end_space:]

    def _evict_oldest_episode(self):
        """
        Remove the oldest stored episode (whole episode), updating ring pointers & meta.
        """
        if not self._episodes_meta:
            return
        ep_len, _, _ = self._episodes_meta.pop(0)
        # advance start and shrink size
        self._start = (self._start + ep_len) % self.max_transitions
        self._size -= ep_len
        if self._size < 0:
            self._size = 0

    def _ensure_capacity_for(self, extra: int):
        """
        Ensure we have room for `extra` more transitions by evicting whole episodes.
        """
        # If episode itself is larger than capacity, keep only the last part of it.
        if extra > self.max_transitions:
            # wipe all existing episodes
            self._episodes_meta.clear()
            self._start = 0
            self._size = 0
            return

        while self._size + extra > self.max_transitions and self._episodes_meta:
            self._evict_oldest_episode()

        # If still too big (shouldn't happen unless meta empty), reset
        if self._size + extra > self.max_transitions and not self._episodes_meta:
            self._start = 0
            self._size = 0

    def _append_episode_tensors(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        next_states: torch.Tensor,
        rewards: torch.Tensor,
    ):
        """
        states/actions/next_states/rewards are (L, dim) tensors on any device.
        We'll store only usable_len = max(L-1, 0) transitions (to mirror CPU code).
        """
        L = int(states.shape[0])
        usable_len = max(L - 1, 0)
        if usable_len == 0:
            return

        # Keep only first usable transitions
        states = states[:usable_len]
        actions = actions[:usable_len]
        next_states = next_states[:usable_len]
        rewards = rewards[:usable_len]

        # If episode too long, keep only the last max_transitions transitions
        if usable_len > self.max_transitions:
            states = states[-self.max_transitions :]
            actions = actions[-self.max_transitions :]
            next_states = next_states[-self.max_transitions :]
            rewards = rewards[-self.max_transitions :]
            usable_len = self.max_transitions
            # wipe everything else (to mirror "ensure capacity" behavior)
            self._episodes_meta.clear()
            self._start = 0
            self._size = 0

        # Make sure storage exists
        self._ensure_allocated(states[0], actions[0])

        # Move to device/dtype once (episode-wise copy is much cheaper than per-step)
        states = states.to(device=self.device, dtype=self.dtype, non_blocking=True)
        actions = actions.to(device=self.device, dtype=self.dtype, non_blocking=True)
        next_states = next_states.to(device=self.device, dtype=self.dtype, non_blocking=True)
        rewards = rewards.to(device=self.device, dtype=self.dtype, non_blocking=True)

        # Ensure capacity and compute write position
        self._ensure_capacity_for(usable_len)
        write_pos = (self._start + self._size) % self.max_transitions

        # Write data
        assert self._states is not None
        assert self._actions is not None
        assert self._next_states is not None
        assert self._rewards is not None

        self._write_block(self._states, states, write_pos)
        self._write_block(self._actions, actions, write_pos)
        self._write_block(self._next_states, next_states, write_pos)

        # rewards to shape (T,1)
        if rewards.ndim == 1:
            rewards = rewards.unsqueeze(-1)
        elif rewards.ndim == 2 and rewards.shape[1] != 1:
            rewards = rewards.reshape(-1, 1)
        self._write_block(self._rewards, rewards, write_pos)

        # Update meta and size (meta values kept as Python floats for cheap stats)
        ep_return = float(rewards.sum().detach().cpu())
        ep_mean_reward = float(rewards.mean().detach().cpu())
        self._episodes_meta.append((usable_len, ep_return, ep_mean_reward))
        self._size += usable_len

    # ---------- Storage ----------

    def store_step(self, state, action, next_state, reward):
        # Keep raw objects; we will bulk-transfer on done_episode()
        self.tmp_episode_buff.append((state, action, next_state, reward))

    def done_episode(self):
        if not self.tmp_episode_buff:
            return
        self.store_episodes([self.tmp_episode_buff])
        self.tmp_episode_buff = []

    def store_episodes(self, new_episodes: Sequence[Sequence[Tuple[Any, Any, Any, Any]]]):
        """
        new_episodes: list of episodes
          Each episode: list of (state, action, next_state, reward)
        """
        for episode in new_episodes:
            if len(episode) == 0:
                continue

            states, actions, next_states, rewards = zip(*episode)

            # Convert to torch tensors on CPU first (cheap), then move once per episode
            states = torch.as_tensor(states, dtype=self.dtype)
            actions = torch.as_tensor(actions, dtype=self.dtype)
            next_states = torch.as_tensor(next_states, dtype=self.dtype)
            rewards = torch.as_tensor(rewards, dtype=self.dtype)

            # Ensure 2D shapes
            if states.ndim == 1:
                states = states.unsqueeze(0)
            if actions.ndim == 1:
                actions = actions.unsqueeze(0)
            if next_states.ndim == 1:
                next_states = next_states.unsqueeze(0)

            # rewards can be (L,) or (L,1)
            if rewards.ndim == 0:
                rewards = rewards.view(1)
            if rewards.ndim == 1:
                rewards = rewards.view(-1, 1)

            self._append_episode_tensors(states, actions, next_states, rewards)

    # ---------- Sampling ----------

    def _sample_indices(self, batch_size: int, replace: bool) -> torch.Tensor:
        buffer_size = len(self)
        if buffer_size == 0:
            raise ValueError("ReplayBuffer is empty.")

        if (not replace) and batch_size > buffer_size:
            replace = True

        if replace:
            return torch.randint(0, buffer_size, (batch_size,), device=self.device)

        # Approx. unique sampling without O(N) randperm:
        # sample extra, unique, top up if needed.
        idx = torch.randint(0, buffer_size, (batch_size * 2,), device=self.device)
        idx = torch.unique(idx)
        while idx.numel() < batch_size:
            extra = torch.randint(0, buffer_size, (batch_size - idx.numel(),), device=self.device)
            idx = torch.unique(torch.cat([idx, extra], dim=0))
        return idx[:batch_size]

    def sample_batch(self, batch_size: int, replace: bool = False):
        idx = self._sample_indices(batch_size, replace)
        return self.batch_get(idx)

    def sample_batch_stacked(self, batch_size: int, replace: bool = False):
        s_chunks, a_chunks, ns_chunks, r_chunks = self.sample_batch(batch_size, replace)
        # In GPU version, chunks are torch tensors on GPU -> use torch.cat
        return (
            torch.cat(s_chunks, dim=0),
            torch.cat(a_chunks, dim=0),
            torch.cat(ns_chunks, dim=0),
            torch.cat(r_chunks, dim=0),
        )

    def batch_get(self, indices: Any):
        """
        Fetch multiple transitions by flattened indices.

        For compatibility with your CPU buffer, returns lists of chunks.
        GPU version returns a single chunk per field (fast path).
        """
        if isinstance(indices, torch.Tensor):
            idx = indices.to(device=self.device, dtype=torch.long)
        else:
            idx = torch.as_tensor(indices, device=self.device, dtype=torch.long)

        buffer_size = len(self)
        if buffer_size == 0:
            raise ValueError("ReplayBuffer is empty.")

        # Map logical index -> storage index in ring
        storage_idx = (idx + self._start) % self.max_transitions

        assert self._states is not None
        assert self._actions is not None
        assert self._next_states is not None
        assert self._rewards is not None

        s = self._states[storage_idx]
        a = self._actions[storage_idx]
        ns = self._next_states[storage_idx]
        r = self._rewards[storage_idx]

        # Return "chunked lists" (single chunk) like CPU buffer's contract
        return [s], [a], [ns], [r]

    # ---------- Basic API ----------

    def __len__(self):
        return self._size

    def __getitem__(self, idx: int):
        if idx < 0 or idx >= len(self):
            raise IndexError("ReplayBuffer index out of range")
        storage_idx = (self._start + int(idx)) % self.max_transitions

        assert self._states is not None
        assert self._actions is not None
        assert self._next_states is not None
        assert self._rewards is not None

        return (
            self._states[storage_idx],
            self._actions[storage_idx],
            self._next_states[storage_idx],
            self._rewards[storage_idx],
        )

    # ---------- Stats ----------

    def mean_reward(self) -> float:
        if not self._episodes_meta:
            return 0.0
        # Mirror CPU: mean over episodes of mean reward
        return float(sum(mr for _, _, mr in self._episodes_meta) / len(self._episodes_meta))

    def mean_return(self) -> float:
        if not self._episodes_meta:
            return 0.0
        # Mirror CPU: mean over episodes of total return
        return float(sum(ret for _, ret, _ in self._episodes_meta) / len(self._episodes_meta))

    def mean_episode_length(self) -> float:
        """
        Average stored episode length in transitions (usable_len = episode_len - 1).
        """
        if not self._episodes_meta:
            return 0.0
        return float(sum(L for L, _, _ in self._episodes_meta) / len(self._episodes_meta))

    # ---------- Save/Load ----------

    def state_dict(self) -> dict:
        """
        Returns a CPU-serializable snapshot.
        Note: stores the flat buffers (only the valid window) for simplicity.
        """
        if len(self) == 0:
            return {
                "max_transitions": self.max_transitions,
                "device": str(self.device),
                "dtype": str(self.dtype),
                "state_dim": self._state_dim,
                "action_dim": self._action_dim,
                "size": 0,
                "episodes_meta": [],
                "data": None,
            }

        # Export the logical window [0..size) in order (CPU tensors)
        idx = torch.arange(self._size, device=self.device, dtype=torch.long)
        storage_idx = (idx + self._start) % self.max_transitions

        data = {
            "states": self._states[storage_idx].detach().cpu(),
            "actions": self._actions[storage_idx].detach().cpu(),
            "next_states": self._next_states[storage_idx].detach().cpu(),
            "rewards": self._rewards[storage_idx].detach().cpu(),
        }

        return {
            "max_transitions": self.max_transitions,
            "device": str(self.device),
            "dtype": str(self.dtype),
            "state_dim": self._state_dim,
            "action_dim": self._action_dim,
            "size": self._size,
            "episodes_meta": list(self._episodes_meta),
            "data": data,
        }

    def load_state_dict(self, data: dict):
        self.max_transitions = int(data["max_transitions"])
        self._state_dim = data.get("state_dim", None)
        self._action_dim = data.get("action_dim", None)
        self._episodes_meta = list(data.get("episodes_meta", []))
        self._start = 0
        self._size = int(data.get("size", 0))

        payload = data.get("data", None)
        if payload is None or self._size == 0:
            self._states = None
            self._actions = None
            self._next_states = None
            self._rewards = None
            return

        # Allocate
        assert self._state_dim is not None and self._action_dim is not None
        N = self.max_transitions
        self._states = torch.empty((N, self._state_dim), device=self.device, dtype=self.dtype)
        self._actions = torch.empty((N, self._action_dim), device=self.device, dtype=self.dtype)
        self._next_states = torch.empty((N, self._state_dim), device=self.device, dtype=self.dtype)
        self._rewards = torch.empty((N, 1), device=self.device, dtype=self.dtype)

        # Load into beginning of ring
        s = payload["states"].to(self.device, dtype=self.dtype)
        a = payload["actions"].to(self.device, dtype=self.dtype)
        ns = payload["next_states"].to(self.device, dtype=self.dtype)
        r = payload["rewards"].to(self.device, dtype=self.dtype)

        T = s.shape[0]
        if T != self._size:
            self._size = int(T)

        self._states[:T] = s
        self._actions[:T] = a
        self._next_states[:T] = ns
        self._rewards[:T] = r


    def store_episode_stacked(self, states, actions, next_states, rewards):
        """
        Store one episode given as stacked arrays/tensors.

        Inputs:
          states:      (L, obs_dim)  numpy or torch (CPU/GPU ok)
          actions:     (L, act_dim)  numpy or torch
          next_states: (L, obs_dim)  numpy or torch
          rewards:     (L,) or (L,1) numpy or torch

        Behavior matches CPU buffer:
          usable_len = max(L - 1, 0)
          only first usable_len transitions are stored.
        """

        # Convert to torch on CPU first (cheap, consistent)
        if isinstance(states, np.ndarray):
            states_t = torch.from_numpy(states)
        else:
            states_t = torch.as_tensor(states)

        if isinstance(actions, np.ndarray):
            actions_t = torch.from_numpy(actions)
        else:
            actions_t = torch.as_tensor(actions)

        if isinstance(next_states, np.ndarray):
            next_states_t = torch.from_numpy(next_states)
        else:
            next_states_t = torch.as_tensor(next_states)

        if isinstance(rewards, np.ndarray):
            rewards_t = torch.from_numpy(rewards)
        else:
            rewards_t = torch.as_tensor(rewards)

        # Ensure float32-ish
        states_t = states_t.to(dtype=self.dtype, device="cpu")
        actions_t = actions_t.to(dtype=self.dtype, device="cpu")
        next_states_t = next_states_t.to(dtype=self.dtype, device="cpu")
        rewards_t = rewards_t.to(dtype=self.dtype, device="cpu")

        # Ensure 2D for states/actions/next_states
        if states_t.ndim == 1:
            states_t = states_t.unsqueeze(0)
        if actions_t.ndim == 1:
            actions_t = actions_t.unsqueeze(0)
        if next_states_t.ndim == 1:
            next_states_t = next_states_t.unsqueeze(0)

        # rewards -> (L,1)
        if rewards_t.ndim == 0:
            rewards_t = rewards_t.view(1, 1)
        elif rewards_t.ndim == 1:
            rewards_t = rewards_t.view(-1, 1)
        elif rewards_t.ndim == 2 and rewards_t.shape[1] != 1:
            rewards_t = rewards_t.reshape(-1, 1)

        L = int(states_t.shape[0])
        usable_len = max(L - 1, 0)
        if usable_len == 0:
            return

        # Mirror CPU logic: keep only first usable_len
        states_t = states_t[:usable_len]
        actions_t = actions_t[:usable_len]
        next_states_t = next_states_t[:usable_len]
        rewards_t = rewards_t[:usable_len]

        # Allocate on first use
        self._ensure_allocated(states_t[0], actions_t[0])

        # If episode too long, keep last max_transitions and wipe old content
        if usable_len > self.max_transitions:
            states_t = states_t[-self.max_transitions:]
            actions_t = actions_t[-self.max_transitions:]
            next_states_t = next_states_t[-self.max_transitions:]
            rewards_t = rewards_t[-self.max_transitions:]
            usable_len = self.max_transitions
            self._episodes_meta.clear()
            self._start = 0
            self._size = 0

        # Ensure capacity by evicting whole episodes
        self._ensure_capacity_for(usable_len)

        # Compute write position in ring
        write_pos = (self._start + self._size) % self.max_transitions

        # Move once per episode to GPU
        states_g = states_t.to(self.device, non_blocking=True)
        actions_g = actions_t.to(self.device, non_blocking=True)
        next_states_g = next_states_t.to(self.device, non_blocking=True)
        rewards_g = rewards_t.to(self.device, non_blocking=True)

        # Write blocks
        self._write_block(self._states, states_g, write_pos)
        self._write_block(self._actions, actions_g, write_pos)
        self._write_block(self._next_states, next_states_g, write_pos)
        self._write_block(self._rewards, rewards_g, write_pos)

        # Update meta + size (store meta as python floats)
        ep_return = float(rewards_t.sum().item())
        ep_mean_reward = float(rewards_t.mean().item())
        self._episodes_meta.append((usable_len, ep_return, ep_mean_reward))
        self._size += usable_len

    
