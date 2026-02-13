import numpy as np
import gymnasium as gym

class InvertedVelocityWrapper(gym.Wrapper):
    def __init__(self, env,  args):
        super().__init__(env)
        self.invert = 1.0
        self.backward_prob = 0.5 
        self.healthy_reward_weight = args.healthy_reward_weight
        self.vel_scale = args.velocity_reward_scale
        self.scale_wrong_direction = args.scale_wrong_direction_reward
        
        
        # NEW: Curriculum Speed Limit
        self.current_max_speed = float('inf') 

        # CRITICAL FIX FOR HALF-CHEETAH / ANT:
        # We must expand the observation space by 1 to hold the "Task Hint"
        # (The +1.0 or -1.0 value that tells the agent which way to go initially)
        low = np.append(env.observation_space.low, -float('inf'))
        high = np.append(env.observation_space.high, float('inf'))
        self.observation_space = gym.spaces.Box(low, high, dtype=np.float64)

    def update_speed_range(self, min_speed, max_speed):
        """Called by the curriculum loop to ramp up difficulty."""
        self.current_max_speed = float(max_speed)

    def set_backward_prob(self, prob):
        self.backward_prob = float(prob)

    def reset(self, seed=None, options=None):
        # 1. EVALUATION OVERRIDE
        if options and 'task_mode' in options:
            self.invert = float(options['task_mode'])
            
        # 2. TRAINING AUTO-SAMPLE (Randomize every episode for single-episode adaptation)
        else:
            is_backward = np.random.random() < self.backward_prob
            self.invert = -1.0 if is_backward else 1.0
            
        obs, info = self.env.reset(seed=seed, options=options)
        
        # Inject Hint
        obs = self._add_hint(obs)
        
        # Add telemetry
        info['task_direction'] = self.invert
        return obs, info
    
    def step(self, action):
        obs, rewards, terminated, truncated, info = self.env.step(action)
        
        # Inject Hint
        obs = self._add_hint(obs)        
        assert ("reward_forward" in info) or ("forward_reward" in info), info.keys()

        reward_forward = info.get("reward_forward", info.get("forward_reward", 0.0))
        others = rewards - reward_forward

        # --- 1. CALCULATE TASK REWARD ---
        flipped_forward_rew = reward_forward * self.invert * self.vel_scale
        if flipped_forward_rew < 0:
            flipped_forward_rew *= self.scale_wrong_direction
       
        
        # --- 3. TOTAL ---
        # Added 'healthy_reward' to the sum (it was missing in your snippet)
        total_reward = others + flipped_forward_rew
        
        # Telemetry
        info['velocity_reward'] = flipped_forward_rew
        info['task_direction'] = self.invert
        
        return obs, total_reward, terminated, truncated, info

    def _add_hint(self, obs):
        """Appends the target direction (+1 or -1) to the observation vector."""
        return np.append(obs, self.invert)


class GoalPositionWrapper(gym.Wrapper):
    """
    Multitask: Ant soll zu einer (x,y)-Zielposition laufen.
    - Ersetzt reward_forward durch Velocity-in-Goal-Richtung
    - Optionaler zusätzlicher Positions-/Progress-Reward
    - Fügt Hint zur Observation hinzu: (dir_x, dir_y, dist)
    """
    def __init__(self, env, args):
        super().__init__(env)

        # Reward scales
        self.vel_scale = args.velocity_reward_scale
        self.pos_scale = args.position_reward_scale
        self.scale_wrong_direction = args.scale_wrong_direction_reward
        self.maximum_area = args.maximum_area

        # Task sampling
        # self.goal_list = self.make_goal_list(args)  # i.e. [(5,0),(-5,0),(0,5),(0,-5)]
        #self.goal_radius = args.goal_radius
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
        #print(x_goal, y_goal)
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
        options_env = dict(options) if options else None
        target_goal = None
        if options_env and "target_goal" in options_env:
            target_goal = options_env.pop("target_goal")
        obs, info = self.env.reset(seed=seed, options=options_env)

        if target_goal is not None:
            self.goal = np.array(target_goal, dtype=np.float64)
        else:
            self.goal = self._sample_goal()

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

