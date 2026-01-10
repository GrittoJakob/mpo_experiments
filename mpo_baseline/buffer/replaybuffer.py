import numpy as np
from typing import Any, List, Sequence, Tuple


class ReplayBuffer:
    """
    Episodic replay buffer with random access by flattened transition index.

    Stores episodes as tuples of numpy arrays:
      (states, actions, next_states, rewards, terminated, truncated)

    Notes:
    - Stores ALL transitions of an episode (no "L-1" trimming).
    - Evicts whole episodes from the front to keep total transitions <= max_transitions.
    - Uses episode start offsets + searchsorted (no huge idx_to_episode_idx list).
    """

    def __init__(self, max_replay_buffer: int = 2_000_000):
        self.max_transitions = int(max_replay_buffer)

        self.episodes: List[Tuple[np.ndarray, ...]] = []
        self.episode_starts: List[int] = []  # start index (flattened) for each episode
        self.total_size: int = 0

        self.tmp_episode_buff: List[Tuple[Any, ...]] = []

    # -------------------- Internal helpers --------------------

    def _rebuild_starts(self) -> None:
        """Recompute episode_starts and total_size from scratch (used after eviction/load)."""
        self.episode_starts = []
        start = 0
        for ep in self.episodes:
            self.episode_starts.append(start)
            start += int(ep[0].shape[0])  # states length = episode length
        self.total_size = start

    def _ensure_capacity(self) -> None:
        """Evict oldest episodes until total_size <= max_transitions."""
        removed = False
        while self.total_size > self.max_transitions and self.episodes:
            # remove oldest episode
            old = self.episodes.pop(0)
            self.total_size -= int(old[0].shape[0])
            removed = True

        if removed:
            self._rebuild_starts()

    @staticmethod
    def _as_2d(x: np.ndarray) -> np.ndarray:
        if x.ndim == 1:
            return x[None, :]
        return x

    @staticmethod
    def _as_col(x: np.ndarray) -> np.ndarray:
        # (L,) -> (L,1), scalar -> (1,1), (L,k) -> reshape to (L,1) if needed
        if x.ndim == 0:
            return x.reshape(1, 1)
        if x.ndim == 1:
            return x.reshape(-1, 1)
        if x.ndim == 2 and x.shape[1] != 1:
            return x.reshape(-1, 1)
        return x

    # -------------------- Storage --------------------

    def store_step(self, state, action, next_state, reward, terminated=False, truncated=False):
        """Optional step-wise API."""
        self.tmp_episode_buff.append((state, action, next_state, reward, terminated, truncated))

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
        """
        for episode in new_episodes:
            if len(episode) == 0:
                continue

            # Unpack (expects 6-tuple per step)
            states, actions, next_states, rewards, terminated, truncated = zip(*episode)

            # Convert to numpy (float32 everywhere for speed/consistency)
            states_np = np.asarray(states, dtype=np.float32)
            actions_np = np.asarray(actions, dtype=np.float32)
            next_states_np = np.asarray(next_states, dtype=np.float32)
            rewards_np = np.asarray(rewards, dtype=np.float32)
            terminated_np = np.asarray(terminated, dtype=np.float32)
            truncated_np = np.asarray(truncated, dtype=np.float32)

            # Ensure shapes
            states_np = self._as_2d(states_np)
            actions_np = self._as_2d(actions_np)
            next_states_np = self._as_2d(next_states_np)

            rewards_np = self._as_col(rewards_np)
            terminated_np = self._as_col(terminated_np)
            truncated_np = self._as_col(truncated_np)

            L = int(states_np.shape[0])
            if L == 0:
                continue

            # Append episode
            self.episode_starts.append(self.total_size)
            self.episodes.append((states_np, actions_np, next_states_np, rewards_np, terminated_np, truncated_np))
            self.total_size += L

        self._ensure_capacity()

    # -------------------- Sampling --------------------

    def __len__(self) -> int:
        return self.total_size

    def _episode_index_for(self, idx: np.ndarray) -> np.ndarray:
        """
        Map flattened indices -> episode indices using searchsorted.
        """
        starts = np.asarray(self.episode_starts, dtype=np.int64)
        # ep = rightmost start <= idx
        ep_idx = np.searchsorted(starts, idx, side="right") - 1
        return ep_idx

    def sample_batch(self, batch_size: int, replace: bool = False):
        buffer_size = len(self)
        if buffer_size == 0:
            raise ValueError("ReplayBuffer is empty.")

        if (not replace) and batch_size > buffer_size:
            replace = True

        idx = np.random.choice(buffer_size, size=batch_size, replace=replace)
        return self.batch_get(idx)

    def sample_batch_stacked(self, batch_size: int, replace: bool = False):
        s, a, ns, r, term, trunc = self.sample_batch(batch_size, replace)
        return (
            np.concatenate(s, axis=0),
            np.concatenate(a, axis=0),
            np.concatenate(ns, axis=0),
            np.concatenate(r, axis=0),
            np.concatenate(term, axis=0),
            np.concatenate(trunc, axis=0),
        )

    def batch_get(self, indices):
        """
        Efficiently fetch multiple transitions by flattened indices.

        Returns lists of numpy arrays (chunked by episode group),
        so caller can concatenate cheaply.
        """
        idx = np.asarray(indices, dtype=np.int64)
        if idx.size == 0:
            return [], [], [], [], [], []

        if idx.min() < 0 or idx.max() >= self.total_size:
            raise IndexError("ReplayBuffer index out of range")

        ep_idx = self._episode_index_for(idx)
        starts = np.asarray(self.episode_starts, dtype=np.int64)
        local_i = idx - starts[ep_idx]

        # group by episode
        order = np.argsort(ep_idx)
        ep_idx = ep_idx[order]
        local_i = local_i[order]

        state_chunks, action_chunks, next_state_chunks = [], [], []
        reward_chunks, term_chunks, trunc_chunks = [], [], []

        unique_eps, starts_pos = np.unique(ep_idx, return_index=True)

        for u, s_pos in enumerate(starts_pos):
            e = int(unique_eps[u])
            end = int(starts_pos[u + 1]) if (u + 1) < len(starts_pos) else len(ep_idx)

            i_list = local_i[s_pos:end]

            states, actions, next_states, rewards, terminated, truncated = self.episodes[e]

            state_chunks.append(states[i_list])
            action_chunks.append(actions[i_list])
            next_state_chunks.append(next_states[i_list])
            reward_chunks.append(rewards[i_list])
            term_chunks.append(terminated[i_list])
            trunc_chunks.append(truncated[i_list])

        return state_chunks, action_chunks, next_state_chunks, reward_chunks, term_chunks, trunc_chunks

    # -------------------- Random access --------------------

    def __getitem__(self, idx: int):
        if idx < 0 or idx >= self.total_size:
            raise IndexError("ReplayBuffer index out of range")

        idx_arr = np.asarray([idx], dtype=np.int64)
        ep = int(self._episode_index_for(idx_arr)[0])
        start = self.episode_starts[ep]
        i = idx - start

        states, actions, next_states, rewards, terminated, truncated = self.episodes[ep]
        return states[i], actions[i], next_states[i], rewards[i], terminated[i], truncated[i]

    # -------------------- Stats --------------------

    def mean_reward(self) -> float:
        if not self.episodes:
            return 0.0
        return float(np.mean([float(ep[3].mean()) for ep in self.episodes]))

    def mean_return(self) -> float:
        if not self.episodes:
            return 0.0
        return float(np.mean([float(ep[3].sum()) for ep in self.episodes]))

    def mean_episode_length(self) -> float:
        if not self.episodes:
            return 0.0
        return float(np.mean([int(ep[0].shape[0]) for ep in self.episodes]))

    # -------------------- Save/Load --------------------

    def state_dict(self) -> dict:
        return {
            "episodes": self.episodes,
            "episode_starts": self.episode_starts,
            "total_size": self.total_size,
            "max_transitions": self.max_transitions,
        }

    def load_state_dict(self, data: dict) -> None:
        self.episodes = data["episodes"]
        self.max_transitions = int(data["max_transitions"])
        # Rebuild is safer than trusting stored starts/size
        self._rebuild_starts()
        self._ensure_capacity()
