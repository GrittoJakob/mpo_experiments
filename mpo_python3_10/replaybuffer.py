import numpy as np


class ReplayBuffer:
    def __init__(self, max_replay_buffer = 200000):

        # buffers
        self.start_idx_of_episode = []
        self.idx_to_episode_idx = []
        self.episodes = []
        self.tmp_episode_buff = []
        self.max_transitions = max_replay_buffer
    """
    def _current_size(self):
        return len(self.idx_to_episode_idx)
    """
    def store_step(self, state, action, next_state, reward):
        self.tmp_episode_buff.append(
            (state, action, next_state, reward))
    
    def _ensure_capacity(self):
        while self._current_size() > self.max_transitions:
            # remove oldest episode
            states, actions, next_states, rewards = self.episodes.pop(0)
            ep_len = len(states) 

            # Move all idx one index before (episode_idx(1) -> episode_idx(0))
            del self.start_idx_of_episode[0]

            # neue Start-Indices berechnen
            for i in range(len(self.start_idx_of_episode)):
                self.start_idx_of_episode[i] -= ep_len

            # 3. idx_to_episode_idx updaten (alle Einträge um 1 nach oben schieben)
            self.idx_to_episode_idx = self.idx_to_episode_idx[ep_len:]
            self.idx_to_episode_idx = [i - 1 for i in self.idx_to_episode_idx]
    """

    def done_episode(self):
        states, actions, next_states, rewards = zip(*self.tmp_episode_buff)
        episode_len = len(states)
        usable_episode_len = episode_len - 1
        self.start_idx_of_episode.append(len(self.idx_to_episode_idx))
        self.idx_to_episode_idx.extend([len(self.episodes)] * usable_episode_len)
        self.episodes.append((states, actions, next_states, rewards))
        self.tmp_episode_buff = []
    """

    def store_episodes(self, new_episodes):
        for episode in new_episodes:
            states, actions, next_states, rewards = zip(*episode)
            episode_len = len(states)
            usable_episode_len = episode_len - 1
            self.start_idx_of_episode.append(len(self.idx_to_episode_idx))
            self.idx_to_episode_idx.extend([len(self.episodes)] * usable_episode_len)
            self.episodes.append((states, actions, next_states, rewards))
        self._ensure_capacity

    def clear(self):
        self.start_idx_of_episode = []
        self.idx_to_episode_idx = []
        self.episodes = []
        self.tmp_episode_buff = []

    def __getitem__(self, idx):
        episode_idx = self.idx_to_episode_idx[idx]
        start_idx = self.start_idx_of_episode[episode_idx]
        i = idx - start_idx
        states, actions, next_states, rewards = self.episodes[episode_idx]
        state, action, next_state, reward = states[i], actions[i], next_states[i], rewards[i]
        return state, action, next_state, reward

    def debug_summary(self):
        print("\n=== Replay Buffer Summary ===")
        print(f"Total transitions stored: {len(self.idx_to_episode_idx)}")
        print(f"Total episodes stored:    {len(self.episodes)}\n")

        for i, (states, actions, next_states, rewards) in enumerate(self.episodes):
            print(f"  Episode {i:3d} | length: {len(states)} transitions | "
                f"start index: {self.start_idx_of_episode[i]}")
        print("================================\n")

    def __len__(self):
        return len(self.idx_to_episode_idx)

    def print_episode(self, ep_idx, max_steps=50):
        """
        Pretty-print a full episode, optionally truncated to max_steps.
        """
        if ep_idx < 0 or ep_idx >= len(self.episodes):
            print(f"Episode {ep_idx} does not exist!")
            return

        states, actions, next_states, rewards = self.episodes[ep_idx]

        print(f"\n=== Episode {ep_idx} (len={len(states)}) ===")
        steps_to_print = min(len(states), max_steps)

        for t in range(steps_to_print):
            print(f"\n-- Step {t} --")
            print("State:      ", np.array(states[t]))
            print("Action:     ", np.array(actions[t]))
            print("Next state: ", np.array(next_states[t]))
            print("Reward:     ", rewards[t])

        if len(states) > max_steps:
            print(f"\n... truncated after {max_steps} steps ...")

        print("==============================\n")

    def mean_reward(self):
        _, _, _, rewards = zip(*self.episodes)
        return np.mean([np.mean(reward) for reward in rewards])

    def mean_return(self):
        _, _, _, rewards = zip(*self.episodes)
        return np.mean([np.sum(reward) for reward in rewards])

    
    def state_dict(self):
        """Return all buffer data as a serializable dictionary."""
        return {
            "episodes": self.episodes,
            "start_idx_of_episode": self.start_idx_of_episode,
            "idx_to_episode_idx": self.idx_to_episode_idx,
            "max_transitions": self.max_transitions
        }

    def load_state_dict(self, data):
        """Load buffer contents from saved dictionary."""
        self.episodes = data["episodes"]
        self.start_idx_of_episode = data["start_idx_of_episode"]
        self.idx_to_episode_idx = data["idx_to_episode_idx"]
        self.max_transitions = data["max_transitions"]