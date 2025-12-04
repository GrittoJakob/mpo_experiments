import numpy as np


class ReplayBuffer:
    """
    Episodic replay buffer with random access by flattened transition index.

    Internally it stores:
    - self.episodes: list of episodes,
      each episode is a tuple (states, actions, next_states, rewards)
      where each of these is a sequence (list/tuple) of length T (number of transitions).
    - self.idx_to_episode_idx: for each flattened transition index,
      which episode it belongs to
    - self.start_idx_of_episode: starting flattened index for each episode
    """

    def __init__(self, max_replay_buffer: int = 2_000_000):
        """
        :param max_replay_buffer: maximum number of transitions stored.
                                  Oldest episodes are removed when exceeded.
        """
        # Episode indexing structures
        self.start_idx_of_episode = []   # list[int], start index (in flat space) of each episode
        self.idx_to_episode_idx = []     # list[int], maps flat index -> episode index
        self.episodes = []               # list[(states, actions, next_states, rewards)]
        self.tmp_episode_buff = []       # optional buffer for step-wise storing

        # Capacity limit in number of transitions
        self.max_transitions = max_replay_buffer

    # ---------- Internal helpers ----------

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
            episode_len = len(states)       # number of transitions in this episode
            usable_episode_len = max(episode_len, 0)   # <-- alle Transitionen nutzen

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

    # ---------- Store interface ----------

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
            usable_episode_len = max(episode_len, 0)   # <-- alle Transitionen nutzen

            # Append episode data
            self.episodes.append((states, actions, next_states, rewards))

            # Record starting flat index for this episode
            self.start_idx_of_episode.append(len(self.idx_to_episode_idx))

            # For each usable transition in this episode, map flat index -> episode index
            self.idx_to_episode_idx.extend([len(self.episodes) - 1] * usable_episode_len)

        # Make sure we respect capacity constraints
        self._ensure_capacity()

    # ---------- Basic API (unchanged für dein MPO) ----------

    def __len__(self) -> int:
        """Number of transitions in the flattened buffer."""
        return len(self.idx_to_episode_idx)

    def __getitem__(self, idx: int):
        """
        Random-access a single transition by its flattened index.

        :param idx: integer in [0, len(self)-1]
        :return: (state, action, next_state, reward)
        """
        episode_idx = self.idx_to_episode_idx[idx]
        start_idx = self.start_idx_of_episode[episode_idx]
        i = idx - start_idx  # index innerhalb der Episode

        states, actions, next_states, rewards = self.episodes[episode_idx]
        state, action, next_state, reward = states[i], actions[i], next_states[i], rewards[i]
        return state, action, next_state, reward

    # ---------- N-Step / Sequenz-API für Retrace ----------

    def get_nstep_transition(self, idx: int, n: int, gamma: float = 1.0):
        """
        Build an n-step transition (s_t, a_t, s_{t+n}, R_t^{(n)}, done) starting
        from a flattened index idx, *without crossing episode boundaries*.

        R_t^{(n)} = sum_{k=0}^{k_end} gamma^k r_{t+k},
        where k_end = min(n-1, T-1-t) and T is episode length.

        :param idx: flattened transition index (0 <= idx < len(self))
        :param n:   number of steps for the return (n-step)
        :param gamma: discount factor
        :return: (state_t, action_t, next_state_last, R_n, done)
                 - state_t:       s_t
                 - action_t:      a_t
                 - next_state_last: s_{t+k_end+1} (i.e., next_state of the last used transition)
                 - R_n:           discounted n-step return starting at t
                 - done:          True if episode ended within these n steps
        """
        episode_idx = self.idx_to_episode_idx[idx]
        start_idx = self.start_idx_of_episode[episode_idx]
        t = idx - start_idx

        states, actions, next_states, rewards = self.episodes[episode_idx]
        T = len(rewards)  # number of transitions in this episode

        # how far we can go inside this episode
        last_t = min(t + n - 1, T - 1)

        # starting state & action
        s_t = states[t]
        a_t = actions[t]

        # n-step discounted return
        R = 0.0
        discount = 1.0
        for k in range(t, last_t + 1):
            R += discount * rewards[k]
            discount *= gamma

        s_last_next = next_states[last_t]
        done = (last_t == T - 1)  # True if we hit end of episode

        return s_t, a_t, s_last_next, R, done

    def get_sequence(self, idx: int, length: int):
        """
        Return a raw sequence of up to `length` consecutive transitions
        within the same episode, starting at flattened index `idx`.

        :return:
            states_seq:      list of states [s_t, s_{t+1}, ..., s_{t+L-1}]
            actions_seq:     list of actions
            next_states_seq: list of next_states
            rewards_seq:     list of rewards
        """
        episode_idx = self.idx_to_episode_idx[idx]
        start_idx = self.start_idx_of_episode[episode_idx]
        t = idx - start_idx

        states, actions, next_states, rewards = self.episodes[episode_idx]
        T = len(rewards)

        last_t = min(t + length - 1, T - 1)

        states_seq = states[t:last_t + 1]
        actions_seq = actions[t:last_t + 1]
        next_states_seq = next_states[t:last_t + 1]
        rewards_seq = rewards[t:last_t + 1]

        return states_seq, actions_seq, next_states_seq, rewards_seq

    # ---------- Statistics ----------

    def mean_reward(self) -> float:
        """
        Average reward per step over all episodes.
        """
        if not self.episodes:
            return 0.0

        _, _, _, rewards = zip(*self.episodes)
        return float(np.mean([np.mean(r) for r in rewards]))

    def mean_return(self) -> float:
        """
        Average return (sum of rewards per episode) over all episodes.
        """
        if not self.episodes:
            return 0.0

        _, _, _, rewards = zip(*self.episodes)
        return float(np.mean([np.sum(r) for r in rewards]))

    # ---------- Save / Load ----------

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