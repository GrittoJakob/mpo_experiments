import numpy as np
import torch
from collections import deque
from typing import Any, Deque, List, Optional, Sequence, Tuple


class ReplayBufferGPU:
    """
    Episodic replay buffer with a flat ring-buffer on GPU.

    Stores transitions:
      (state, action, next_state, reward, terminated, truncated)

    Key points:
    - Stores full episodes (all transitions, no "L-1" trimming).
    - Evicts whole episodes (FIFO) to respect max_transitions.
    - Sampling is uniform over stored transitions.
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

        # Lazy init shapes
        self._state_dim: Optional[int] = None
        self._action_dim: Optional[int] = None

        # Flat ring buffers (allocated lazily)
        self._states: Optional[torch.Tensor] = None
        self._actions: Optional[torch.Tensor] = None
        self._next_states: Optional[torch.Tensor] = None
        self._rewards: Optional[torch.Tensor] = None          # (N, 1)
        self._terminated: Optional[torch.Tensor] = None       # (N, 1) float32 0/1
        self._truncated: Optional[torch.Tensor] = None        # (N, 1) float32 0/1
        self._task_invert: Optional[torch.Tensor] = None
        self._vel_rew: Optional[torch.Tensor] = None


        # Ring pointers: logical window is [start .. start+size) modulo N
        self._start: int = 0
        self._size: int = 0

        # Episode FIFO metadata: (length, return_sum, mean_reward)
        self._episodes: Deque[Tuple[int, float, float]] = deque()

        # Optional step-wise API (kept for compatibility)
        self.tmp_episode_buff: List[Tuple[Any, Any, Any, Any, Any, Any, Any, Any]] = []

    # -------------------- Internal helpers --------------------

    def _ensure_allocated(self, state_example, action_example) -> None:
        if self._states is not None:
            return

        s = torch.as_tensor(state_example).reshape(-1)
        a = torch.as_tensor(action_example).reshape(-1)
        self._state_dim = int(s.numel())
        self._action_dim = int(a.numel())

        N, sd, ad = self.max_transitions, self._state_dim, self._action_dim

        self._states = torch.empty((N, sd), device=self.device, dtype=self.dtype)
        self._actions = torch.empty((N, ad), device=self.device, dtype=self.dtype)
        self._next_states = torch.empty((N, sd), device=self.device, dtype=self.dtype)
        self._rewards = torch.empty((N, 1), device=self.device, dtype=self.dtype)
        self._terminated = torch.empty((N, 1), device=self.device, dtype=self.dtype)
        self._truncated = torch.empty((N, 1), device=self.device, dtype=self.dtype)
        self._task_invert = torch.empty((N,1), device = self.device, dtype = self.dtype)

    @staticmethod
    def _to_2d(x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 1:
            return x.unsqueeze(0)
        return x

    @staticmethod
    def _to_col(x: torch.Tensor) -> torch.Tensor:
        # shape -> (L,1)
        if x.ndim == 0:
            return x.view(1, 1)
        if x.ndim == 1:
            return x.view(-1, 1)
        if x.ndim == 2 and x.shape[1] != 1:
            return x.reshape(-1, 1)
        return x

    def _write_block(self, buf: torch.Tensor, data: torch.Tensor, write_pos: int) -> None:
        """Write (T,D) data into ring buffer starting at write_pos (mod N)."""
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

    def _evict_one_episode(self) -> None:
        """Evict the oldest episode (FIFO)."""
        if not self._episodes:
            return
        ep_len, _, _ = self._episodes.popleft()
        self._start = (self._start + ep_len) % self.max_transitions
        self._size -= ep_len
        if self._size < 0:
            self._size = 0

    def _ensure_capacity_for(self, extra: int) -> None:
        """Evict episodes until we have room for `extra` transitions."""
        if extra <= 0:
            return

        # If an incoming episode is longer than total capacity: keep only its last part.
        if extra >= self.max_transitions:
            self._episodes.clear()
            self._start = 0
            self._size = 0
            return

        while self._size + extra > self.max_transitions and self._episodes:
            self._evict_one_episode()

        # If still too big (shouldn't happen often), hard reset
        if self._size + extra > self.max_transitions and not self._episodes:
            self._start = 0
            self._size = 0

    def _append_episode(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        next_states: torch.Tensor,
        rewards: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
        task_invert: torch.Tensor,
        vel_rew: torch.Tensor,
    ) -> None:
        """
        Append one episode (L transitions) into the GPU ring buffer.
        Tensors can be on CPU or GPU; we copy episode-wise once.
        """
        L = int(states.shape[0])
        if L <= 0:
            return

        # If episode too long, keep only the last (max_transitions-1) or so is handled by _ensure_capacity_for.
        if L >= self.max_transitions:
            # Keep only last max_transitions-1 transitions (since extra==max_transitions triggers reset above)
            # More simply: keep last (max_transitions - 1) transitions to satisfy extra < max_transitions.
            keep = self.max_transitions - 1
            states = states[-keep:]
            actions = actions[-keep:]
            next_states = next_states[-keep:]
            rewards = rewards[-keep:]
            terminated = terminated[-keep:]
            truncated = truncated[-keep:]
            task_invert = task_invert[-keep:]
            vel_rew = vel_rew[-keep:]
            L = keep

        # Allocate on first use
        self._ensure_allocated(states[0], actions[0])

        # Ensure capacity (evict whole episodes)
        self._ensure_capacity_for(L)
        write_pos = (self._start + self._size) % self.max_transitions

        # Move once per episode to GPU
        states_g = states.to(self.device, dtype=self.dtype, non_blocking=True)
        actions_g = actions.to(self.device, dtype=self.dtype, non_blocking=True)
        next_states_g = next_states.to(self.device, dtype=self.dtype, non_blocking=True)
        rewards_g = rewards.to(self.device, dtype=self.dtype, non_blocking=True)
        terminated_g = terminated.to(self.device, dtype=self.dtype, non_blocking=True)
        truncated_g = truncated.to(self.device, dtype=self.dtype, non_blocking=True)
        task_invert_g = task_invert.to(self.device, dtype=self.dtpye, non_blocking= True)
        vel_rew_g = vel_rew_g.to(self.device, dtpye=self.dtype, non_blocking=True)

        assert self._states is not None
        assert self._actions is not None
        assert self._next_states is not None
        assert self._rewards is not None
        assert self._terminated is not None
        assert self._truncated is not None
        assert self._task_invert is not None
        assert self._vel_rew is not None

        self._write_block(self._states, states_g, write_pos)
        self._write_block(self._actions, actions_g, write_pos)
        self._write_block(self._next_states, next_states_g, write_pos)
        self._write_block(self._rewards, rewards_g, write_pos)
        self._write_block(self._terminated, terminated_g, write_pos)
        self._write_block(self._truncated, truncated_g, write_pos)
        self._write_block(self._task_invert, task_invert_g, write_pos)
        self._write_block(self._vel_rew, vel_rew_g, write_pos)


        ep_return = float(rewards.sum().detach().cpu())
        ep_mean_reward = float(rewards.mean().detach().cpu())
        ep_vel_return = float(vel_rew.sum().detach().cpu())
        ep_vel_reward = float(vel_rew.mean().detach().cpu())
        self._episodes.append((L, ep_return, ep_mean_reward, ep_vel_return, ep_vel_reward))
        self._size += L

    # -------------------- Storage API --------------------

    def store_step(self, state, action, next_state, reward, terminated=False, truncated=False, task_invert= 1.0, vel_rew=0.0):
        """Optional step-wise API (kept for compatibility)."""
        self.tmp_episode_buff.append((state, action, next_state, reward, terminated, truncated, task_invert, vel_rew))

    def done_episode(self):
        if not self.tmp_episode_buff:
            return
        self.store_episodes([self.tmp_episode_buff])
        self.tmp_episode_buff = []

    def store_episodes(self, new_episodes: Sequence[Sequence[Tuple[Any, ...]]]) -> None:
        """
        new_episodes: list of episodes
          Each episode: list of tuples:
            (state, action, next_state, reward, terminated, truncated)
          For backwards compatibility, 4-tuples are accepted and treated as terminated=0, truncated=0.
        """
        for episode in new_episodes:
            if len(episode) == 0:
                continue

            # Backward compatible unpacking
            if len(episode[0]) == 4:
                states, actions, next_states, rewards = zip(*episode)
                terminated = [0.0] * len(rewards)
                truncated = [0.0] * len(rewards)
            else:
                states, actions, next_states, rewards, terminated, truncated, task_invert, vel_rew = zip(*episode)

            # Convert to CPU tensors first
            states_t = self._to_2d(torch.as_tensor(states, dtype=self.dtype, device="cpu"))
            actions_t = self._to_2d(torch.as_tensor(actions, dtype=self.dtype, device="cpu"))
            next_states_t = self._to_2d(torch.as_tensor(next_states, dtype=self.dtype, device="cpu"))

            rewards_t = self._to_col(torch.as_tensor(rewards, dtype=self.dtype, device="cpu"))
            terminated_t = self._to_col(torch.as_tensor(terminated, dtype=self.dtype, device="cpu"))
            truncated_t = self._to_col(torch.as_tensor(truncated, dtype=self.dtype, device="cpu"))

            task_invert_t = self._to_col(torch.as_tensor(task_invert, dtype= self.dtype, device="cpu"))
            vel_rew_t = self._to_col(torch.as_tensor(vel_rew, dtype= self.dtype, device="cpu"))
            self._append_episode(states_t, actions_t, next_states_t, rewards_t, terminated_t, truncated_t, task_invert_t, vel_rew_t)

    def store_episode_stacked(
        self,
        states,
        actions,
        next_states,
        rewards,
        terminated=None,
        truncated=None,
        task_invert=1.0,
        vel_rew=0.0,
    ) -> None:
        """
        Store one episode as stacked arrays/tensors.

        Required shapes:
          states:      (L, obs_dim)
          actions:     (L, act_dim)
          next_states: (L, obs_dim)
          rewards:     (L,) or (L,1)

        Optional:
          terminated:  (L,) or (L,1)  (float/bool)
          truncated:   (L,) or (L,1)  (float/bool)
        """

        def to_cpu_tensor(x) -> torch.Tensor:
            if isinstance(x, np.ndarray):
                t = torch.from_numpy(x)
            else:
                t = torch.as_tensor(x)
            return t.to(device="cpu", dtype=self.dtype)

        states_t = self._to_2d(to_cpu_tensor(states))
        actions_t = self._to_2d(to_cpu_tensor(actions))
        next_states_t = self._to_2d(to_cpu_tensor(next_states))
        rewards_t = self._to_col(to_cpu_tensor(rewards))
        task_invert_t= self._to_col(to_cpu_tensor(task_invert))
        vel_rew_t = self._to_colU(to_cpu_tensor(vel_rew))

        L = int(states_t.shape[0])
        if terminated is None:
            terminated_t = torch.zeros((L, 1), dtype=self.dtype, device="cpu")
        else:
            terminated_t = self._to_col(to_cpu_tensor(terminated))

        if truncated is None:
            truncated_t = torch.zeros((L, 1), dtype=self.dtype, device="cpu")
        else:
            truncated_t = self._to_col(to_cpu_tensor(truncated))

        self._append_episode(states_t, actions_t, next_states_t, rewards_t, terminated_t, truncated_t, task_invert_t, vel_rew_t)

    # -------------------- Sampling API --------------------

    def __len__(self) -> int:
        return self._size

    def _sample_indices(self, batch_size: int, replace: bool) -> torch.Tensor:
        if self._size <= 0:
            raise ValueError("ReplayBuffer is empty.")

        if (not replace) and batch_size > self._size:
            replace = True

        if replace:
            return torch.randint(0, self._size, (batch_size,), device=self.device)

        # Unique sampling without randperm (approximate, but fine)
        idx = torch.randint(0, self._size, (batch_size * 2,), device=self.device)
        idx = torch.unique(idx)
        while idx.numel() < batch_size:
            extra = torch.randint(0, self._size, (batch_size - idx.numel(),), device=self.device)
            idx = torch.unique(torch.cat([idx, extra], dim=0))
        return idx[:batch_size]

    def batch_get(self, indices: Any):
        """
        For compatibility with the CPU buffer, returns chunk-lists (single chunk each).
        """
        if isinstance(indices, torch.Tensor):
            idx = indices.to(device=self.device, dtype=torch.long)
        else:
            idx = torch.as_tensor(indices, device=self.device, dtype=torch.long)

        if self._size <= 0:
            raise ValueError("ReplayBuffer is empty.")

        storage_idx = (idx + self._start) % self.max_transitions

        assert self._states is not None
        assert self._actions is not None
        assert self._next_states is not None
        assert self._rewards is not None
        assert self._terminated is not None
        assert self._truncated is not None
        assert self._task_invert is not None
        assert self._vel_rew is not None

        state = self._states[storage_idx]
        action = self._actions[storage_idx]
        next_state = self._next_states[storage_idx]
        rew = self._rewards[storage_idx]
        term = self._terminated[storage_idx]
        trunc = self._truncated[storage_idx]
        # task_invert = self._task_invert[storage_idx]
        # vel_rew = self._vel_rew[storage_idx]


        return [state], [action], [next_state], [rew], [term], [trunc]

    def sample_batch(self, batch_size: int, replace: bool = False):
        idx = self._sample_indices(batch_size, replace)
        return self.batch_get(idx)

    def sample_batch_stacked(self, batch_size: int, replace: bool = False):
        s, a, ns, r, term, trunc = self.sample_batch(batch_size, replace)
        # single chunk -> just return that tensor
        return (
            torch.cat(s, dim=0),
            torch.cat(a, dim=0),
            torch.cat(ns, dim=0),
            torch.cat(r, dim=0),
            torch.cat(term, dim=0),
            torch.cat(trunc, dim=0),
        )

    def __getitem__(self, idx: int):
        if idx < 0 or idx >= self._size:
            raise IndexError("ReplayBuffer index out of range")

        storage_idx = (self._start + int(idx)) % self.max_transitions

        assert self._states is not None
        assert self._actions is not None
        assert self._next_states is not None
        assert self._rewards is not None
        assert self._terminated is not None
        assert self._truncated is not None

        return (
            self._states[storage_idx],
            self._actions[storage_idx],
            self._next_states[storage_idx],
            self._rewards[storage_idx],
            self._terminated[storage_idx],
            self._truncated[storage_idx],
        )

    # -------------------- Stats --------------------

    def mean_reward(self) -> float:
        if not self._episodes:
            return 0.0
        return float(sum(mr for _, _, mr in self._episodes) / len(self._episodes))

    def mean_return(self) -> float:
        if not self._episodes:
            return 0.0
        return float(sum(ret for _, ret, _ in self._episodes) / len(self._episodes))

    def mean_episode_length(self) -> float:
        if not self._episodes:
            return 0.0
        return float(sum(L for L, _, _ in self._episodes) / len(self._episodes))
    def mean_vel_rew(self) -> float:

    # -------------------- Save/Load --------------------

    def state_dict(self) -> dict:
        """
        CPU-serializable snapshot of the logical window [0..size) in order.
        """
        if self._size == 0:
            return {
                "max_transitions": self.max_transitions,
                "device": str(self.device),
                "dtype": str(self.dtype),
                "state_dim": self._state_dim,
                "action_dim": self._action_dim,
                "size": 0,
                "episodes": list(self._episodes),
                "data": None,
            }

        idx = torch.arange(self._size, device=self.device, dtype=torch.long)
        storage_idx = (idx + self._start) % self.max_transitions

        assert self._states is not None
        assert self._actions is not None
        assert self._next_states is not None
        assert self._rewards is not None
        assert self._terminated is not None
        assert self._truncated is not None

        data = {
            "states": self._states[storage_idx].detach().cpu(),
            "actions": self._actions[storage_idx].detach().cpu(),
            "next_states": self._next_states[storage_idx].detach().cpu(),
            "rewards": self._rewards[storage_idx].detach().cpu(),
            "terminated": self._terminated[storage_idx].detach().cpu(),
            "truncated": self._truncated[storage_idx].detach().cpu(),
        }

        return {
            "max_transitions": self.max_transitions,
            "device": str(self.device),
            "dtype": str(self.dtype),
            "state_dim": self._state_dim,
            "action_dim": self._action_dim,
            "size": self._size,
            "episodes": list(self._episodes),
            "data": data,
        }

    def load_state_dict(self, data: dict) -> None:
        self.max_transitions = int(data["max_transitions"])
        self._state_dim = data.get("state_dim", None)
        self._action_dim = data.get("action_dim", None)

        self._episodes = deque(data.get("episodes", []))
        self._start = 0
        self._size = int(data.get("size", 0))

        payload = data.get("data", None)
        if payload is None or self._size == 0:
            self._states = None
            self._actions = None
            self._next_states = None
            self._rewards = None
            self._terminated = None
            self._truncated = None
            return

        assert self._state_dim is not None and self._action_dim is not None

        N, sd, ad = self.max_transitions, self._state_dim, self._action_dim
        self._states = torch.empty((N, sd), device=self.device, dtype=self.dtype)
        self._actions = torch.empty((N, ad), device=self.device, dtype=self.dtype)
        self._next_states = torch.empty((N, sd), device=self.device, dtype=self.dtype)
        self._rewards = torch.empty((N, 1), device=self.device, dtype=self.dtype)
        self._terminated = torch.empty((N, 1), device=self.device, dtype=self.dtype)
        self._truncated = torch.empty((N, 1), device=self.device, dtype=self.dtype)

        s = payload["states"].to(self.device, dtype=self.dtype)
        a = payload["actions"].to(self.device, dtype=self.dtype)
        ns = payload["next_states"].to(self.device, dtype=self.dtype)
        r = payload["rewards"].to(self.device, dtype=self.dtype)
        term = payload.get("terminated", torch.zeros_like(r)).to(self.device, dtype=self.dtype)
        trunc = payload.get("truncated", torch.zeros_like(r)).to(self.device, dtype=self.dtype)

        T = s.shape[0]
        self._size = int(T)

        self._states[:T] = s
        self._actions[:T] = a
        self._next_states[:T] = ns
        self._rewards[:T] = r
        self._terminated[:T] = term
        self._truncated[:T] = trunc
