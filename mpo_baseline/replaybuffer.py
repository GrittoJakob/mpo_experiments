import numpy as np


class ReplayBuffer:
    """
    Episodic replay buffer with random access by flattened transition index.

    Stores episodes as:
      self.episodes: list of tuples (states, actions, next_states, rewards)
                     where each element is a numpy array.
    """

    def __init__(self, max_replay_buffer: int = 2_000_000):
        self.start_idx_of_episode = []
        self.idx_to_episode_idx = []
        self.episodes = []
        self.tmp_episode_buff = []

        self.max_transitions = max_replay_buffer

    # ---------- Internal helpers ----------

    def _current_size(self) -> int:
        return len(self.idx_to_episode_idx)

    def _rebuild_indices(self):
        self.start_idx_of_episode = []
        self.idx_to_episode_idx = []

        for ep_idx, (states, actions, next_states, rewards) in enumerate(self.episodes):
            episode_len = len(states)
            usable_episode_len = max(episode_len - 1, 0)

            self.start_idx_of_episode.append(len(self.idx_to_episode_idx))
            if usable_episode_len > 0:
                self.idx_to_episode_idx.extend([ep_idx] * usable_episode_len)

    def _ensure_capacity(self):
        removed = False
        while self._current_size() > self.max_transitions and self.episodes:
            self.episodes.pop(0)
            removed = True

        if removed:
            self._rebuild_indices()

    # ---------- Storage ----------

    def store_step(self, state, action, next_state, reward):
        self.tmp_episode_buff.append((state, action, next_state, reward))

    def done_episode(self):
        if not self.tmp_episode_buff:
            return

        self.store_episodes([self.tmp_episode_buff])
        self.tmp_episode_buff = []

    def store_episodes(self, new_episodes):
        """
        new_episodes: list of episodes
          Each episode: list of (state, action, next_state, reward)
        """
        for episode in new_episodes:
            if len(episode) == 0:
                continue

            states, actions, next_states, rewards = zip(*episode)

            # Convert to numpy arrays for faster batch access later
            states = np.asarray(states)
            actions = np.asarray(actions)
            next_states = np.asarray(next_states)
            rewards = np.asarray(rewards)

            episode_len = len(states)
            usable_episode_len = max(episode_len - 1, 0)

            self.episodes.append((states, actions, next_states, rewards))

            # Update indices incrementally
            self.start_idx_of_episode.append(len(self.idx_to_episode_idx))
            if usable_episode_len > 0:
                self.idx_to_episode_idx.extend([len(self.episodes) - 1] * usable_episode_len)

        self._ensure_capacity()

    # ---------- Sampling ----------

    def sample_batch(self, batch_size: int, replace: bool = False):
        buffer_size = len(self)
        if buffer_size == 0:
            raise ValueError("ReplayBuffer is empty.")

        if (not replace) and batch_size > buffer_size:
            replace = True

        indices = np.random.choice(buffer_size, size=batch_size, replace=replace)
        return self.batch_get(indices)

    def sample_batch_stacked(self, batch_size: int, replace: bool = False):
        s, a, ns, r = self.sample_batch(batch_size, replace)
        return np.concatenate(s, axis=0), np.concatenate(a, axis=0), \
               np.concatenate(ns, axis=0), np.concatenate(r, axis=0)

    def batch_get(self, indices):
        """
        Efficiently fetch multiple transitions by flattened indices.

        Returns lists of numpy arrays (chunked by episode group),
        so caller can np.concatenate/stack cheaply.
        """
        idx = np.asarray(indices, dtype=np.int64)

        idx_to_ep = np.asarray(self.idx_to_episode_idx, dtype=np.int64)
        start_idx = np.asarray(self.start_idx_of_episode, dtype=np.int64)

        ep_idx = idx_to_ep[idx]
        ep_start = start_idx[ep_idx]
        local_i = idx - ep_start

        # group by episode
        order = np.argsort(ep_idx)
        ep_idx = ep_idx[order]
        local_i = local_i[order]

        state_chunks = []
        action_chunks = []
        next_state_chunks = []
        reward_chunks = []

        unique_eps, starts = np.unique(ep_idx, return_index=True)

        for u, s in enumerate(starts):
            e = unique_eps[u]
            end = starts[u + 1] if u + 1 < len(starts) else len(ep_idx)

            i_list = local_i[s:end]

            states, actions, next_states, rewards = self.episodes[e]

            # Fancy indexing -> much faster than Python loop
            state_chunks.append(states[i_list])
            action_chunks.append(actions[i_list])
            next_state_chunks.append(next_states[i_list])
            reward_chunks.append(rewards[i_list])

        return state_chunks, action_chunks, next_state_chunks, reward_chunks

    # ---------- Basic API ----------

    def __len__(self):
        return len(self.idx_to_episode_idx)

    def __getitem__(self, idx: int):
        episode_idx = self.idx_to_episode_idx[idx]
        start_idx = self.start_idx_of_episode[episode_idx]
        i = idx - start_idx

        states, actions, next_states, rewards = self.episodes[episode_idx]
        return states[i], actions[i], next_states[i], rewards[i]

    # ---------- Stats ----------

    def mean_reward(self) -> float:
        if not self.episodes:
            return 0.0
        rewards = [ep[3] for ep in self.episodes]
        return float(np.mean([np.mean(r) for r in rewards]))

    def mean_return(self) -> float:
        if not self.episodes:
            return 0.0
        rewards = [ep[3] for ep in self.episodes]
        return float(np.mean([np.sum(r) for r in rewards]))

    # ---------- Save/Load ----------

    def state_dict(self) -> dict:
        return {
            "episodes": self.episodes,
            "start_idx_of_episode": self.start_idx_of_episode,
            "idx_to_episode_idx": self.idx_to_episode_idx,
            "max_transitions": self.max_transitions,
        }

    def load_state_dict(self, data: dict):
        self.episodes = data["episodes"]
        self.start_idx_of_episode = data["start_idx_of_episode"]
        self.idx_to_episode_idx = data["idx_to_episode_idx"]
        self.max_transitions = data["max_transitions"]








# import numpy as np


# class ReplayBuffer:
#     """
#     Episodic replay buffer with random access by flattened transition index.

#     Internally it stores:
#     - self.episodes: list of episodes, each episode is a list of (s, a, s', r)
#     - self.idx_to_episode_idx: for each flattened transition index, which episode it belongs to
#     - self.start_idx_of_episode: starting flattened index for each episode
#     """   

#     def __init__(self, max_replay_buffer: int = 2000000):
#         """
#         :param max_replay_buffer: maximum number of transitions stored.
#                                   Oldest episodes are removed when exceeded.
#         """
#         # Episode indexing structures
#         self.start_idx_of_episode = []   # list[int], start index (in flat space) of each episode
#         self.idx_to_episode_idx = []     # list[int], maps flat index -> episode index
#         self.episodes = []               # list[list[(s, a, s', r)]]
#         self.tmp_episode_buff = []       # optional buffer for step-wise storing (currently unused)

#         # Capacity limit in number of transitions
#         self.max_transitions = max_replay_buffer

#     # Internal helpers
#     def _current_size(self) -> int:
#         """Number of valid transitions currently stored (flattened)."""
#         return len(self.idx_to_episode_idx)
    
#     def _rebuild_indices(self):
#         """
#         Rebuild start_idx_of_episode and idx_to_episode_idx based on self.episodes.
#         This is called after removing old episodes to keep the indexing consistent.
#         """

#         self.start_idx_of_episode = []
#         self.idx_to_episode_idx = []

#         for ep_idx, (states, actions, next_states, rewards) in enumerate(self.episodes):
#             episode_len = len(states)
            
#             # Usable transitions per episode: typically episode_len - 1
#             usable_episode_len = max(episode_len - 1, 0)

#             # Starting flat index of this episode
#             self.start_idx_of_episode.append(len(self.idx_to_episode_idx))

#             # For each usable transition in this episode, record which episode it belongs to
#             self.idx_to_episode_idx.extend([ep_idx] * usable_episode_len)

#     def _ensure_capacity(self):
#         """
#         Ensure the buffer does not exceed max_transitions.

#         If too many transitions are stored, remove oldest episodes until
#         the number of transitions is <= max_transitions.
#         Then rebuild the indexing data structures.
#         """
#         # Remove episodes from the front until we are under capacity
#         removed = False
#         while self._current_size() > self.max_transitions and len(self.episodes) > 0:
#             # Remove oldest episode (index 0)
#             self.episodes.pop(0)
#             removed = True

#         if removed:
#             # Rebuild indices from the remaining episodes
#             self._rebuild_indices()
        

#     def store_step(self, state, action, next_state, reward):
#         """
#         Store a single transition in a temporary episode buffer.
#         This is useful if you want to collect one episode step-by-step and
#         then call done_episode() at the end. 
#         """
#         self.tmp_episode_buff.append((state, action, next_state, reward))

#     def done_episode(self):
#         """
#         Finalize the current tmp_episode_buff as a new episode and add it to the buffer.

#         This is an alternative interface to store_episodes(), suitable when
#         you are collecting transitions one by one.
#         """
#         if not self.tmp_episode_buff:
#             return

#         states, actions, next_states, rewards = zip(*self.tmp_episode_buff)
#         episode = list(zip(states, actions, next_states, rewards))

#         # Append single episode using the same logic as in store_episodes()
#         self.store_episodes([episode])

#         # Clear temporary buffer
#         self.tmp_episode_buff = []

#     def sample_batch(self, batch_size: int, replace: bool = False):
#         """
#         Sample a minibatch of transitions.

#         Returns:
#             state_batch, action_batch, next_state_batch, reward_batch (lists)
#             -> same content as repeatedly calling __getitem__, but faster.
#         """
#         buffer_size = len(self)
#         if buffer_size == 0:
#             raise ValueError("ReplayBuffer is empty.")

#         if (not replace) and batch_size > buffer_size:
#             # fall back to replace to avoid crash in early training
#             replace = True

#         indices = np.random.choice(buffer_size, size=batch_size, replace=replace)
#         return self.batch_get(indices)

#     def batch_get(self, indices):
#         """
#         Efficiently fetch multiple transitions by flattened indices.
#         indices: array-like of shape (B,)

#         Returns:
#             state_batch, action_batch, next_state_batch, reward_batch (lists)
#         """
#         idx = np.asarray(indices, dtype=np.int64)

#         # Convert mapping lists to numpy for fast vectorized indexing
#         idx_to_ep = np.asarray(self.idx_to_episode_idx, dtype=np.int64)
#         start_idx = np.asarray(self.start_idx_of_episode, dtype=np.int64)

#         ep_idx = idx_to_ep[idx]              # (B,)
#         ep_start = start_idx[ep_idx]         # (B,)
#         local_i = idx - ep_start             # (B,)

#         # Group by episode to reduce Python overhead
#         order = np.argsort(ep_idx)
#         ep_idx = ep_idx[order]
#         local_i = local_i[order]

#         state_batch = []
#         action_batch = []
#         next_state_batch = []
#         reward_batch = []

#         # Iterate grouped
#         unique_eps, starts = np.unique(ep_idx, return_index=True)

#         for u, s in enumerate(starts):
#             e = unique_eps[u]
#             end = starts[u + 1] if u + 1 < len(starts) else len(ep_idx)

#             # indices for this episode
#             i_list = local_i[s:end]

#             states, actions, next_states, rewards = self.episodes[e]

#             # pull the transitions
#             for i in i_list:
#                 state_batch.append(states[i])
#                 action_batch.append(actions[i])
#                 next_state_batch.append(next_states[i])
#                 reward_batch.append(rewards[i])

#         return state_batch, action_batch, next_state_batch, reward_batch
    

#     def store_episodes(self, new_episodes):
#         """
#         Store a list of episodes.

#         :param new_episodes: list of episodes, each episode is a list of (s, a, s', r) tuples.
#         """
#         for episode in new_episodes:
#             # Episode is a list of (state, action, next_state, reward)
#             states, actions, next_states, rewards = zip(*episode)
#             episode_len = len(states)
#             usable_episode_len = max(episode_len - 1, 0)

#             # Append episode data
#             self.episodes.append((states, actions, next_states, rewards))

#             # Record starting flat index for this episode
#             self.start_idx_of_episode.append(len(self.idx_to_episode_idx))

#             # For each usable transition in this episode, map flat index -> episode index
#             self.idx_to_episode_idx.extend([len(self.episodes) - 1] * usable_episode_len)

#         # Make sure we respect capacity constraints
#         self._ensure_capacity()

#     # Basic API
#     def __len__(self) -> int:
#         """Number of transitions in the flattened buffer."""
#         return len(self.idx_to_episode_idx)

#     def __getitem__(self, idx: int):
#         """
#         Random-access a transition by its flattened index.

#         :param idx: integer in [0, len(self)-1]
#         :return: (state, action, next_state, reward)
#         """
#         episode_idx = self.idx_to_episode_idx[idx]
#         start_idx = self.start_idx_of_episode[episode_idx]
#         i = idx - start_idx

#         states, actions, next_states, rewards = self.episodes[episode_idx]
#         state, action, next_state, reward = states[i], actions[i], next_states[i], rewards[i]
#         return state, action, next_state, reward
    
#     # Statistics
#     def mean_reward(self) -> float:
#         """
#         Average reward per step over all episodes.

#         :return: scalar float, 0.0 if buffer is empty.
#         """
#         if not self.episodes:
#             return 0.0

#         _, _, _, rewards = zip(*self.episodes)  # rewards is a tuple of sequences (one per episode)
#         return float(np.mean([np.mean(r) for r in rewards]))

#     def mean_return(self) -> float:
#         """
#         Average return (sum of rewards per episode) over all episodes.

#         :return: scalar float, 0.0 if buffer is empty.
#         """
#         if not self.episodes:
#             return 0.0

#         _, _, _, rewards = zip(*self.episodes)
#         return float(np.mean([np.sum(r) for r in rewards]))
    
#     def state_dict(self) -> dict:
#         """Return all buffer data as a serializable dictionary."""
#         return {
#             "episodes": self.episodes,
#             "start_idx_of_episode": self.start_idx_of_episode,
#             "idx_to_episode_idx": self.idx_to_episode_idx,
#             "max_transitions": self.max_transitions,
#         }

#     def load_state_dict(self, data: dict):
#         """Load buffer contents from a saved dictionary."""
#         self.episodes = data["episodes"]
#         self.start_idx_of_episode = data["start_idx_of_episode"]
#         self.idx_to_episode_idx = data["idx_to_episode_idx"]
#         self.max_transitions = data["max_transitions"]


