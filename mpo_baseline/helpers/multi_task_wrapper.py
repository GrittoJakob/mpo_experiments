import gymnasium as gym
import numpy as np
from collections import deque
import gymnasium as gym
import numpy as np
from collections import deque

class Multi_Task_InvertedWrapper(gym.Wrapper):
    def __init__(self, env, args, history_len=5, store_task_reward=False):
        super().__init__(env)

        self.invert = 1.0
        self.backward_prob = 0.5

        self.vel_scale = args.velocity_reward_scale
        self.scale_wrong_direction = args.scale_wrong_direction_reward

        # 🟢 History config
        self.history_len = int(history_len)
        self.store_task_reward = bool(store_task_reward)

        # Action dim (flatten)
        assert isinstance(self.action_space, gym.spaces.Box)
        self.act_dim = int(np.prod(self.action_space.shape))

        # History buffers
        self._hist_rewards = deque(maxlen=self.history_len)
        self._hist_actions = deque(maxlen=self.history_len)
        self._hist_valid  = deque(maxlen=self.history_len)   

        # 🟢 Observation Space: env_obs + H rewards + H*act_dim actions + H valid mask
        obs_low = env.observation_space.low
        obs_high = env.observation_space.high

        # rewards: unbounded
        rew_low = np.full((self.history_len,), -np.inf, dtype=np.float64)
        rew_high = np.full((self.history_len,),  np.inf, dtype=np.float64)

        # actions: bounds repeated H times
        act_low = np.tile(self.action_space.low.flatten(), self.history_len).astype(np.float64)
        act_high = np.tile(self.action_space.high.flatten(), self.history_len).astype(np.float64)

        # valid mask: [0,1] (float)
        valid_low = np.zeros((self.history_len,), dtype=np.float64)
        valid_high = np.ones((self.history_len,), dtype=np.float64)

        low = np.concatenate([obs_low.astype(np.float64), rew_low, act_low, valid_low])
        high = np.concatenate([obs_high.astype(np.float64), rew_high, act_high, valid_high])

        self.observation_space = gym.spaces.Box(low, high, dtype=np.float64)

    def set_backward_prob(self, prob):
        self.backward_prob = float(prob)

    def reset(self, seed=None, options=None):
        # 1) EVAL override
        if options and 'task_mode' in options:
            self.invert = float(options['task_mode'])
        else:
            is_backward = np.random.random() < self.backward_prob
            self.invert = -1.0 if is_backward else 1.0

        obs, info = self.env.reset(seed=seed, options=options)

        # 🟢 History initialisieren: valid=0
        self._hist_rewards.clear()
        self._hist_actions.clear()
        self._hist_valid.clear()
        for _ in range(self.history_len):
            self._hist_rewards.append(0.0)
            self._hist_actions.append(np.zeros((self.act_dim,), dtype=np.float64))
            self._hist_valid.append(0.0)  

        obs = self._augment_obs(obs)

        info['task_direction'] = self.invert
        return obs, info

    def step(self, action):
        obs, rewards, terminated, truncated, info = self.env.step(action)

        # Forward/velocity reward aus Mujoco-infos ziehen
        assert ("reward_forward" in info) or ("forward_reward" in info), info.keys()
        reward_forward = info.get("reward_forward", info.get("forward_reward", 0.0))

        others = rewards - reward_forward

        # Task reward (inverted velocity)
        flipped_forward_rew = reward_forward * self.invert * self.vel_scale
        if flipped_forward_rew < 0:
            flipped_forward_rew *= self.scale_wrong_direction

        total_reward = others + flipped_forward_rew

        # 🟢 History updaten
        action_flat = np.asarray(action, dtype=np.float64).reshape(-1)
        rew_to_store = float(flipped_forward_rew if self.store_task_reward else total_reward)

        self._hist_actions.append(action_flat)
        self._hist_rewards.append(rew_to_store)
        self._hist_valid.append(1.0)  

        obs = self._augment_obs(obs)

        # Telemetry
        info['velocity_reward'] = float(flipped_forward_rew)
        info['task_direction'] = float(self.invert)
        info['history_reward_stored'] = rew_to_store

        return obs, float(total_reward), terminated, truncated, info

    def _augment_obs(self, obs):
        hist_rewards = np.asarray(self._hist_rewards, dtype=np.float64)  # (H,)
        hist_actions = np.asarray(self._hist_actions, dtype=np.float64).reshape(self.history_len * self.act_dim)
        hist_valid  = np.asarray(self._hist_valid, dtype=np.float64)     

        return np.concatenate([np.asarray(obs, dtype=np.float64), hist_rewards, hist_actions, hist_valid])
