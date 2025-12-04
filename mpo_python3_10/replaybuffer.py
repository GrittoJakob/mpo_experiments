import numpy as np


class ReplayBuffer:
    """
    Episodic replay buffer with random access by flattened transition index.

    Internally it stores:
    - self.episodes: list of episodes, each episode is a list of (s, a, s', r)
    - self.idx_to_episode_idx: for each flattened transition index, which episode it belongs to
    - self.start_idx_of_episode: starting flattened index for each episode
    """   

    def __init__(self, max_replay_buffer: int = 200000):
        """
        :param max_replay_buffer: maximum number of transitions stored.
                                  Oldest episodes are removed when exceeded.
        """
        # Episode indexing structures
        self.start_idx_of_episode = []   # list[int], start index (in flat space) of each episode
        self.idx_to_episode_idx = []     # list[int], maps flat index -> episode index
        self.episodes = []               # list[list[(s, a, s', r)]]
        self.tmp_episode_buff = []       # optional buffer for step-wise storing (currently unused)

        # Capacity limit in number of transitions
        self.max_transitions = max_replay_buffer

    # Internal helpers
    def _current_size(self) -> int:
        """Number of valid transitions currently stored (flattened)."""
        return len(self.idx_to_episode_idx)
    
    def _rebuild_indices(self):
        """
        Rebuild start_idx_of_episode and idx_to_episode_idx based on self.episodes.
        This is called after removing old episodes to keep the indexing consistent.
        """

        self.start_idx_of_episode = []
        self.idx_to_episode_idx = []

        for ep_idx, (states, actions, next_states, rewards) in enumerate(self.episodes):
            episode_len = len(states)
            
            # Usable transitions per episode: typically episode_len - 1
            usable_episode_len = max(episode_len - 1, 0)

            # Starting flat index of this episode
            self.start_idx_of_episode.append(len(self.idx_to_episode_idx))

            # For each usable transition in this episode, record which episode it belongs to
            self.idx_to_episode_idx.extend([ep_idx] * usable_episode_len)

    def _ensure_capacity(self):
        """
        Ensure the buffer does not exceed max_transitions.

        If too many transitions are stored, remove oldest episodes until
        the number of transitions is <= max_transitions.
        Then rebuild the indexing data structures.
        """
        # Remove episodes from the front until we are under capacity
        while self._current_size() > self.max_transitions and len(self.episodes) > 0:
            # Remove oldest episode (index 0)
            self.episodes.pop(0)

        # Rebuild indices from the remaining episodes
        self._rebuild_indices()

    def store_step(self, state, action, next_state, reward):
        """
        Store a single transition in a temporary episode buffer.
        This is useful if you want to collect one episode step-by-step and
        then call done_episode() at the end. 
        """
        self.tmp_episode_buff.append((state, action, next_state, reward))

    def done_episode(self):
        """
        Finalize the current tmp_episode_buff as a new episode and add it to the buffer.

        This is an alternative interface to store_episodes(), suitable when
        you are collecting transitions one by one.
        """
        if not self.tmp_episode_buff:
            return

        states, actions, next_states, rewards = zip(*self.tmp_episode_buff)
        episode = list(zip(states, actions, next_states, rewards))

        # Append single episode using the same logic as in store_episodes()
        self.store_episodes([episode])

        # Clear temporary buffer
        self.tmp_episode_buff = []

    def store_episodes(self, new_episodes):
        """
        Store a list of episodes.

        :param new_episodes: list of episodes, each episode is a list of (s, a, s', r) tuples.
        """
        for episode in new_episodes:
            # Episode is a list of (state, action, next_state, reward)
            states, actions, next_states, rewards = zip(*episode)
            episode_len = len(states)
            usable_episode_len = max(episode_len - 1, 0)

            # Append episode data
            self.episodes.append((states, actions, next_states, rewards))

            # Record starting flat index for this episode
            self.start_idx_of_episode.append(len(self.idx_to_episode_idx))

            # For each usable transition in this episode, map flat index -> episode index
            self.idx_to_episode_idx.extend([len(self.episodes) - 1] * usable_episode_len)

        # Make sure we respect capacity constraints
        self._ensure_capacity()

    # Basic API
    def __len__(self) -> int:
        """Number of transitions in the flattened buffer."""
        return len(self.idx_to_episode_idx)

    def __getitem__(self, idx: int):
        """
        Random-access a transition by its flattened index.

        :param idx: integer in [0, len(self)-1]
        :return: (state, action, next_state, reward)
        """
        episode_idx = self.idx_to_episode_idx[idx]
        start_idx = self.start_idx_of_episode[episode_idx]
        i = idx - start_idx

        states, actions, next_states, rewards = self.episodes[episode_idx]
        state, action, next_state, reward = states[i], actions[i], next_states[i], rewards[i]
        return state, action, next_state, reward
    
    # Statistics
    def mean_reward(self) -> float:
        """
        Average reward per step over all episodes.

        :return: scalar float, 0.0 if buffer is empty.
        """
        if not self.episodes:
            return 0.0

        _, _, _, rewards = zip(*self.episodes)  # rewards is a tuple of sequences (one per episode)
        return float(np.mean([np.mean(r) for r in rewards]))

    def mean_return(self) -> float:
        """
        Average return (sum of rewards per episode) over all episodes.

        :return: scalar float, 0.0 if buffer is empty.
        """
        if not self.episodes:
            return 0.0

        _, _, _, rewards = zip(*self.episodes)
        return float(np.mean([np.sum(r) for r in rewards]))
    
    def state_dict(self) -> dict:
        """Return all buffer data as a serializable dictionary."""
        return {
            "episodes": self.episodes,
            "start_idx_of_episode": self.start_idx_of_episode,
            "idx_to_episode_idx": self.idx_to_episode_idx,
            "max_transitions": self.max_transitions,
        }

    def load_state_dict(self, data: dict):
        """Load buffer contents from a saved dictionary."""
        self.episodes = data["episodes"]
        self.start_idx_of_episode = data["start_idx_of_episode"]
        self.idx_to_episode_idx = data["idx_to_episode_idx"]
        self.max_transitions = data["max_transitions"]

