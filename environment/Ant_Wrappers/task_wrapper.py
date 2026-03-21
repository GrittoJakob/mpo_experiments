import numpy as np
import gymnasium as gym

class GoalPositionWrapper(gym.Wrapper):
    """
    Multitask: Ant should move to a target position (x, y).
    - Replaces reward_forward with velocity in the direction of the goal
    - Optional additional position/progress reward
    - Adds a hint to the observation: (dir_x, dir_y, dist)
    """
    def __init__(self, env, args):
        super().__init__(env)

        # Reward scales
        self.vel_scale = args.velocity_reward_scale
        self.pos_scale = args.position_reward_scale
        self.scale_wrong_direction = args.scale_wrong_direction_reward
        self.maximum_area = args.maximum_area
        self.success_radius = args.success_radius

        # Internal state
        self.goal = np.zeros(2, dtype=np.float64)
        self.prev_xy_pos = None

        # Obs: append 3 dims (dir_x, dir_y, dist)
        low = np.append(env.observation_space.low, [-np.inf, -np.inf, 0.0])
        high = np.append(env.observation_space.high, [ np.inf,  np.inf, np.inf])
        self.observation_space = gym.spaces.Box(low, high, dtype=np.float64)

    def _sample_goal(self):
        max_area = self.maximum_area
        
        x_goal = x = np.random.uniform(-max_area, max_area)
        y_goal = x = np.random.uniform(-max_area, max_area)
        return np.array([x_goal, y_goal], dtype=np.float64)

    def _get_xy_from_info(self, info):
        return np.array([info["x_position"], info["y_position"]], dtype=np.float64)

    def _dt(self):      
        dt = getattr(self.env.unwrapped, "dt", None)
        if dt is None:
            dt = 0.05
        return float(dt)

    def _hint(self, xy):
        vec = self.goal - xy
        dist = np.linalg.norm(vec)
        if dist > 1e-8:
            dir_vec = vec / dist
        else:
            dir_vec = np.zeros(2, dtype=np.float64)
        return np.array([dir_vec[0], dir_vec[1], dist], dtype=np.float64)

    def reset(self, seed=None, options=None):
        
        # Check for options in env reset
        options_env = dict(options) if options else None
        target_goal = None
        if options_env and "target_goal" in options_env:
            target_goal = options_env.pop("target_goal")
        obs, info = self.env.reset(seed=seed, options=options_env)

        # Check existing target goal
        if target_goal is not None:
            self.goal = np.array(target_goal, dtype=np.float64)
        else:
            self.goal = self._sample_goal()

        # Get info of position
        xy_pos= self._get_xy_from_info(info)
        self.prev_xy_pos = xy_pos.copy()

        obs = np.append(obs, self._hint(xy_pos))

        info["goal_x"] = float(self.goal[0])
        info["goal_y"] = float(self.goal[1])
        info["distance_to_goal"] = float(np.linalg.norm(self.goal - xy_pos))
        return obs, info

    def step(self, action):
        # Save prev pos
        prev_xy_pos = self.prev_xy_pos.copy()

        obs_raw, reward, terminated, truncated, info = self.env.step(action)

        xy_pos = self._get_xy_from_info(info)
        self.prev_xy_pos = xy_pos.copy()

        # Inject hint
        obs = np.append(obs_raw, self._hint(xy_pos))

        # split base reward and replace it by new reward for position and velocitiy
        reward_forward = info.get("reward_forward", 0.0)
        base_reward = reward - reward_forward  

        # Velecotiy reward
        dt = self._dt()
        vel_xy = (xy_pos - prev_xy_pos) / max(dt, 1e-8)

        dist_to_goal = self.goal - prev_xy_pos
        dist_prev = np.linalg.norm(dist_to_goal)
        if dist_prev > 1e-8:
            u = dist_to_goal / dist_prev
        else:
            u = np.zeros(2, dtype=np.float64)

        v_towards = float(np.dot(vel_xy, u))

        vel_rew = self.vel_scale * v_towards
        if vel_rew < 0:
            vel_rew *= self.scale_wrong_direction

        dist_now = float(np.linalg.norm(self.goal - xy_pos))
        progress = float(dist_prev - dist_now)  # >0 wenn näher gekommen

        pos_rew = self.pos_scale * progress

        # --- Success bonus / termination (optional) ---
        reached = dist_now < self.success_radius
        # Wenn du willst:
        if reached:
            pos_rew += 10.0

            self.goal = self._sample_goal()
            obs =  np.append(obs_raw, self._hint(xy_pos))


        total_reward = base_reward + vel_rew + pos_rew

        # Telemetry
        info["goal_x"] = float(self.goal[0])
        info["goal_y"] = float(self.goal[1])
        info["distance_to_goal"] = dist_now
        info["goal_progress"] = progress
        info["goal_velocity"] = v_towards
        info["velocity_reward"] = vel_rew
        info["position_reward"] = pos_rew
        info["reached_goal"] = reached

        return obs, total_reward, terminated, truncated, info

