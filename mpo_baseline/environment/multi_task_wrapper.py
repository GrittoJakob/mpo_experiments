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
        unwrapped_env = self.env.unwrapped
        spawn_yaw = unwrapped_env.np_random.uniform(-np.pi, np.pi)

        qpos = unwrapped_env.data.qpos.copy()
        qvel = unwrapped_env.data.qvel.copy()

        # Quaternion für Rotation um z (Yaw): (w,x,y,z) = (cos(y/2), 0, 0, sin(y/2))
        root_quaternion = np.array(
            [np.cos(spawn_yaw / 2.0), 0.0, 0.0, np.sin(spawn_yaw / 2.0)],
            dtype=np.float64
        )

        # root free joint: qpos[0:3]=pos, qpos[3:7]=quat
        qpos[3:7] = root_quaternion

        unwrapped_env.set_state(qpos, qvel)
        obs = unwrapped_env._get_obs()  

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

class GoalPositionWrapper_wo_Task_Hint(gym.Wrapper):
    """
    Multitask Target-Goals (Ant v5):
    - ersetzt reward_forward durch Velocity-in-Goal-Richtung (+ optional Progress-Reward)
    - KEIN Task-Hint mehr in der Observation
    - Stattdessen: Rewards + Actions der letzten H Schritte (default H=5) + valid-mask

    Sampling:
    - Goal wird immer mit "neuer Richtung" (gegenüber letztem Goal-Dir) gesampled
      (min_angle per args.min_goal_angle, sonst Default).
    """

    def __init__(self, env, args, history_len=5, store_task_reward=False):
        super().__init__(env)

        # Reward scales
        self.vel_scale = float(args.velocity_reward_scale)
        self.pos_scale = float(args.position_reward_scale)
        self.scale_wrong_direction = float(args.scale_wrong_direction_reward)

        # Goal sampling config
        self.maximum_area = float(args.maximum_area)      # hier als max. Radius um aktuellen xy interpretiert
        self.success_radius = float(args.success_radius)

        # "immer neue Richtung"
        self.min_goal_angle = float(getattr(args, "min_goal_angle", np.pi / 3.0))  # Default: 60°
        self.min_goal_dist = float(getattr(args, "min_goal_dist", 0.0))            # optional

        # History config
        self.history_len = int(history_len)
        self.store_task_reward = bool(store_task_reward)

        # Action dim (flatten)
        assert isinstance(self.action_space, gym.spaces.Box)
        self.act_dim = int(np.prod(self.action_space.shape))

        # History buffers
        self._hist_rewards = deque(maxlen=self.history_len)
        self._hist_actions = deque(maxlen=self.history_len)
        self._hist_valid = deque(maxlen=self.history_len)

        # Internal state
        self.goal = np.zeros(2, dtype=np.float64)
        self.prev_xy_pos = None
        self._last_goal_dir = None  # unit vector

        # Observation space: env_obs + H rewards + H*act_dim actions + H valid mask
        obs_low = env.observation_space.low.astype(np.float64)
        obs_high = env.observation_space.high.astype(np.float64)

        rew_low = np.full((self.history_len,), -np.inf, dtype=np.float64)
        rew_high = np.full((self.history_len,), np.inf, dtype=np.float64)

        act_low = np.tile(self.action_space.low.flatten(), self.history_len).astype(np.float64)
        act_high = np.tile(self.action_space.high.flatten(), self.history_len).astype(np.float64)

        valid_low = np.zeros((self.history_len,), dtype=np.float64)
        valid_high = np.ones((self.history_len,), dtype=np.float64)

        low = np.concatenate([obs_low, rew_low, act_low, valid_low])
        high = np.concatenate([obs_high, rew_high, act_high, valid_high])
        self.observation_space = gym.spaces.Box(low, high, dtype=np.float64)

    def _dt(self):
        dt = getattr(self.env.unwrapped, "dt", None)
        if dt is None:
            dt = 0.05
        return float(dt)

    def _get_xy_from_info(self, info):
        return np.array([info["x_position"], info["y_position"]], dtype=np.float64)

    def _sample_goal(self, xy_now):
        """
        Sample goal als Offset um aktuellen xy_now:
        goal = xy_now + r*[cos(a), sin(a)]
        mit Richtungswechsel gegenüber letztem Goal.
        """
        max_r = max(self.maximum_area, 1e-8)
        min_r = np.clip(self.min_goal_dist, 0.0, max_r)

        # Cos-Schwelle für min_angle
        cos_thr = float(np.cos(self.min_goal_angle))

        for _ in range(200):
            angle = np.random.uniform(-np.pi, np.pi)
            r = np.random.uniform(min_r, max_r)
            dir_vec = np.array([np.cos(angle), np.sin(angle)], dtype=np.float64)

            # neue Richtung erzwingen
            if self._last_goal_dir is not None:
                if float(np.dot(dir_vec, self._last_goal_dir)) > cos_thr:
                    continue  # zu ähnlich -> resample

            goal = xy_now + r * dir_vec
            self._last_goal_dir = dir_vec.copy()
            return goal.astype(np.float64)

        # Fallback (falls es aus irgendeinem Grund nicht klappt)
        angle = np.random.uniform(-np.pi, np.pi)
        dir_vec = np.array([np.cos(angle), np.sin(angle)], dtype=np.float64)
        goal = xy_now + max_r * dir_vec
        self._last_goal_dir = dir_vec.copy()
        return goal.astype(np.float64)

    def _augment_obs(self, obs):
        hist_rewards = np.asarray(self._hist_rewards, dtype=np.float64)  # (H,)
        hist_actions = np.asarray(self._hist_actions, dtype=np.float64).reshape(self.history_len * self.act_dim)
        hist_valid = np.asarray(self._hist_valid, dtype=np.float64)      # (H,)
        return np.concatenate([np.asarray(obs, dtype=np.float64), hist_rewards, hist_actions, hist_valid])

    def reset(self, seed=None, options=None):
        options_env = dict(options) if options else None
        target_goal = None
        if options_env and "target_goal" in options_env:
            target_goal = options_env.pop("target_goal")

        obs, info = self.env.reset(seed=seed, options=options_env)

        xy_pos = self._get_xy_from_info(info)
        self.prev_xy_pos = xy_pos.copy()

        # Goal setzen
        if target_goal is not None:
            self.goal = np.array(target_goal, dtype=np.float64)
            vec = self.goal - xy_pos
            n = np.linalg.norm(vec)
            self._last_goal_dir = (vec / n) if n > 1e-8 else None
        else:
            self.goal = self._sample_goal(xy_pos)

        # History initialisieren: valid=0
        self._hist_rewards.clear()
        self._hist_actions.clear()
        self._hist_valid.clear()
        for _ in range(self.history_len):
            self._hist_rewards.append(0.0)
            self._hist_actions.append(np.zeros((self.act_dim,), dtype=np.float64))
            self._hist_valid.append(0.0)

        obs = self._augment_obs(obs)

        # Telemetry
        info["goal_x"] = float(self.goal[0])
        info["goal_y"] = float(self.goal[1])
        info["distance_to_goal"] = float(np.linalg.norm(self.goal - xy_pos))
        return obs, info

    def step(self, action):
        prev_xy_pos = self.prev_xy_pos.copy()

        obs_raw, reward, terminated, truncated, info = self.env.step(action)

        xy_pos = self._get_xy_from_info(info)
        self.prev_xy_pos = xy_pos.copy()

        # split base reward and replace reward_forward
        reward_forward = info.get("reward_forward", info.get("forward_reward", 0.0))
        base_reward = float(reward - reward_forward)

        # Velocity towards goal
        dt = self._dt()
        vel_xy = (xy_pos - prev_xy_pos) / max(dt, 1e-8)

        to_goal_prev = self.goal - prev_xy_pos
        dist_prev = float(np.linalg.norm(to_goal_prev))
        u = (to_goal_prev / dist_prev) if dist_prev > 1e-8 else np.zeros(2, dtype=np.float64)

        v_towards = float(np.dot(vel_xy, u))
        vel_rew = float(self.vel_scale * v_towards)
        if vel_rew < 0:
            vel_rew *= float(self.scale_wrong_direction)

        dist_now = float(np.linalg.norm(self.goal - xy_pos))
        progress = float(dist_prev - dist_now)  # >0 wenn näher gekommen
        pos_rew = float(self.pos_scale * progress)

        # Success -> Bonus + neues Goal (neue Richtung)
        reached = dist_now < self.success_radius
        if reached:
            pos_rew += 10.0
            self.goal = self._sample_goal(xy_pos)
            dist_after = float(np.linalg.norm(self.goal - xy_pos))
        else:
            dist_after = dist_now

        task_reward = vel_rew + pos_rew
        total_reward = base_reward + task_reward

        # History updaten
        action_flat = np.asarray(action, dtype=np.float64).reshape(-1)
        rew_to_store = float(task_reward if self.store_task_reward else total_reward)

        self._hist_actions.append(action_flat)
        self._hist_rewards.append(rew_to_store)
        self._hist_valid.append(1.0)

        obs = self._augment_obs(obs_raw)

        # Telemetry
        info["goal_x"] = float(self.goal[0])
        info["goal_y"] = float(self.goal[1])
        info["distance_to_goal"] = float(dist_after)
        info["goal_progress"] = float(progress)
        info["goal_velocity"] = float(v_towards)
        info["velocity_reward"] = float(vel_rew)
        info["position_reward"] = float(pos_rew)
        info["task_reward"] = float(task_reward)
        info["reached_goal"] = bool(reached)
        info["history_reward_stored"] = float(rew_to_store)

        return obs, float(total_reward), terminated, truncated, info
