import gymnasium as gym
import numpy as np
import logging
from collections import deque
from scipy.spatial.transform import Rotation

class GoalPositionWrapper_Ferdinand(gym.Wrapper):
    """
    Ant must run to a dynamic (x,y) goal.

    - Task hint (same as colleague): [cos(diff_heading), sin(diff_heading), norm_dist]
      where norm_dist = 1 / (1 + 0.1 * dist)

    - Reward (same as colleague, but without curriculum):
      total = scaled_others + vel_rew + tilt_pen + move_bonus + success_bonus + death_penalty
      (scaled_others defaults to 1.0 scaling, configurable via args.healthy_scale)
    """

    def __init__(self, env, args, log_level=logging.ERROR):
        super().__init__(env)

        # --- logging (optional) ---
        self.logger = logging.getLogger("GoalWrapper")
        if not self.logger.handlers:
            h = logging.StreamHandler()
            h.setFormatter(logging.Formatter("[GoalWrapper] %(message)s"))
            self.logger.addHandler(h)
        self.logger.setLevel(log_level)

        self.spawn_noise_scale = getattr(args, "spawn_noise_scale", 0.1)          # joint noise
        self.spawn_vel_noise_scale = getattr(args, "spawn_vel_noise_scale", 0.1)  # qvel noise
        self.randomize_yaw = getattr(args, "spawn_randomize_yaw", True)
        self.current_max_yaw = float(getattr(args, "spawn_max_yaw", np.pi))         
        self.spawn_settle_steps = int(getattr(args, "spawn_settle_steps", 3))
        self.spawn_height = float(getattr(args, "spawn_height", 0.80))
        self._rng = np.random.default_rng()

        # --- 1. CONFIGURATION ---
        self.vel_scale = args.velocity_reward_scale
        self.pos_scale = args.position_reward_scale
        self.scale_wrong_direction = args.scale_wrong_direction_reward
        self.movement_bonus_scale = args.movement_bonus_scale
        self.death_penalty = args.death_penalty
        self.vel_rew_max_speed = float(getattr(args, "vel_rew_max_speed", 4.0))  # m/s cap für v_towards
        
        self.maximum_area = args.maximum_area
        self.success_radius = args.success_radius
        self.current_max_speed = float('inf')

        # --- 2. CURRICULUM DEFAULTS ---
        self.current_move_bonus = 0.5 
        self.current_tilt_pen = 0.0     
        self.current_healthy_scale = 1.0

        # Target Configs
        self.start_move_bonus = 0.5; self.end_move_bonus = 0.05
        self.start_tilt_pen = 0.0;   self.end_tilt_pen = 0.5
        self.start_healthy_scale = 1.0; self.end_healthy_scale = 2.0

        self.tilt_deadzone = 0.26 
        
        # Internal State
        self.goal = np.zeros(2, dtype=np.float64)
        self.prev_xy_pos = None
        self.goal_queue = []
        self.is_random_wandering = True 

        # --- 3. OBSERVATION SPACE ---
        low = np.append(env.observation_space.low, [-1.0, -1.0, 0.0])
        high = np.append(env.observation_space.high, [ 1.0,  1.0, 1.0])
        self.observation_space = gym.spaces.Box(low, high, dtype=np.float64)

    # 🟢 FAILSAFE: RUNTIME GOAL ADDITION
    def add_goal(self, x, y):
        """Called by the user/demo to inject a goal at runtime."""
        # 1. Validation Check
        if x is None or y is None:
             self.logger.error("⚠️ add_goal received None values! Ignoring.")
             return
        if not (np.isfinite(x) and np.isfinite(y)):
             self.logger.error(f"⚠️ add_goal received non-finite values ({x}, {y})! Ignoring.")
             return

        target = np.array([x, y], dtype=np.float64)
        
        if not self.goal_queue and self.is_random_wandering:
            self.logger.info(f"✅ Immediate Goal Set: ({x:.1f}, {y:.1f}) (Overwrote Random Wandering)")
            self.goal = target
            self.is_random_wandering = False 
        else:
            self.logger.info(f"queueing Goal: ({x:.1f}, {y:.1f}) | Queue Size: {len(self.goal_queue) + 1}")
            self.goal_queue.append(target)

    # def set_friction(self, friction: float):
    #     """
    #     Called via env.call('set_friction', value) to modify floor friction.
    #     1.0 = Normal, 0.1 = Ice, 2.0 = Sticky
    #     """
    #     try:
    #         # We need to dig down to the MuJoCo model
    #         # .unwrapped gives us the base Ant-v4/v5 env
    #         if hasattr(self.env.unwrapped, "model"):
    #             model = self.env.unwrapped.model
                
    #             # geom_friction is shape (n_geoms, 3) -> [slide, spin, roll]
    #             # We overwrite ALL geoms (floor + robot parts) to the new value
    #             # This ensures the interaction between feet and floor changes.
    #             model.geom_friction[:] = friction
                

    def _sample_goal(self):
        x = np.random.uniform(-self.maximum_area, self.maximum_area)
        y = np.random.uniform(-self.maximum_area, self.maximum_area)
        self.is_random_wandering = True 
        return np.array([x, y], dtype=np.float64)

    def _get_xy_from_info(self, info):
        # 1. Try getting from dict first (Fastest/Standard)
        if "x_position" in info and "y_position" in info:
            return np.array([info["x_position"], info["y_position"]], dtype=np.float64)

        # 2. Fallback: Direct MuJoCo Access (Robust)
        # If GoalWrapper runs before PhysicsWrapper, 'info' is empty.
        # We fetch coordinates directly from the simulation data.
        try:
            qpos = self.env.unwrapped.data.qpos
            x, y = qpos[0], qpos[1]
            
            # Optional: Backfill info so other wrappers don't panic
            info["x_position"] = x
            info["y_position"] = y
            
            return np.array([x, y], dtype=np.float64)
            
        except Exception as e:
            # 3. Absolute Failure (Should never happen in Mujoco)
            self.logger.warning(f"⚠️ Could not find Position: {e}. Defaulting to (0,0).")
            return np.array([0.0, 0.0], dtype=np.float64)

    def _get_obs_components_old(self, obs, xy_pos):
        quat = obs[1:5]
        # Failsafe for all-zero quaternion (rare physics explosion)
        if np.allclose(quat, 0):
             self.logger.critical("☢️ Zero Quaternion detected! Physics likely exploded.")
             quat = np.array([1.0, 0.0, 0.0, 0.0]) # Valid Identity

        r = Rotation.from_quat(quat, scalar_first=True)
        yaw, pitch, roll = r.as_euler('zyx')
        
        vec_to_goal = self.goal - xy_pos
        dist = np.linalg.norm(vec_to_goal)
        norm_dist = 1.0 / (1.0 + 0.1 * dist)
        
        global_angle = np.arctan2(vec_to_goal[1], vec_to_goal[0])
        diff = global_angle - yaw
        
        hint = np.array([np.cos(diff), np.sin(diff), norm_dist], dtype=np.float64)
        tilt = np.abs(pitch) + np.abs(roll)
        
        return hint, tilt, dist

    def _get_obs_components(self, obs, xy_pos):
        # --- 1. Quaternion & Failsafe ---
        quat = obs[1:5]
        if np.allclose(quat, 0):
             quat = np.array([1.0, 0.0, 0.0, 0.0])

        r_robot = Rotation.from_quat(quat, scalar_first=True)

        # --- 2. Calculate Terrain Normal (Same as before) ---
        target_up = np.array([0.0, 0.0, 1.0])
        if hasattr(self.env.unwrapped, "model"):
            try:
                model = self.env.unwrapped.model
                floor_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "floor_body")
                if floor_body_id != -1:
                    floor_quat = model.body_quat[floor_body_id]
                    scipy_quat = [floor_quat[1], floor_quat[2], floor_quat[3], floor_quat[0]]
                    target_up = Rotation.from_quat(scipy_quat).apply([0.0, 0.0, 1.0])
            except:
                pass

        # --- 3. Calculate Tilt (Relative to Floor) ---
        robot_z = r_robot.apply([0.0, 0.0, 1.0])
        dot = np.clip(np.dot(robot_z, target_up), -1.0, 1.0)
        current_tilt = np.arccos(dot)

        # --- 4. 🟢 ROBUST HEADING CALCULATION (Vector Projection) ---
        # A. Get Vector to Goal
        vec_to_goal = self.goal - xy_pos
        dist = np.linalg.norm(vec_to_goal)
        
        # B. Get Robot's Global Forward Vector
        # Local X-axis is [1, 0, 0] (Nose of the Ant)
        global_forward = r_robot.apply([1.0, 0.0, 0.0])
        
        # C. Project both to 2D Plane (The "Map")
        # We simply ignore the Z-component
        flat_forward = global_forward[:2]
        flat_goal = vec_to_goal # Already 2D
        
        # D. Calculate Angle Difference
        # arctan2(y, x) gives the global angle of the vector
        global_angle_robot = np.arctan2(flat_forward[1], flat_forward[0])
        global_angle_goal  = np.arctan2(flat_goal[1], flat_goal[0])
        
        diff = global_angle_goal - global_angle_robot
        
        # Normalize diff to [-pi, pi]
        diff = (diff + np.pi) % (2 * np.pi) - np.pi

        # --- 5. Final Output ---
        norm_dist = 1.0 / (1.0 + 0.1 * dist)
        hint = np.array([np.cos(diff), np.sin(diff), norm_dist], dtype=np.float64)
        
        return hint, current_tilt, dist

    def reset(self, seed=None, options=None):
        options_env = dict(options) if options else None
        
        self.goal_queue = []
        self.is_random_wandering = True

        if options_env and "goal_series" in options_env:
            self.goal_queue = list(options_env.pop("goal_series"))
            self.is_random_wandering = False
            
        target_goal = options_env.pop("target_goal") if (options_env and "target_goal" in options_env) else None
        
        obs, info = self.env.reset(seed=seed, options=options_env)
        
        # >>> Random spawn pose (yaw/joints/vel) <<<
        new_obs = self._apply_random_spawn_pose(seed=seed)
        if new_obs is not None:
            obs = new_obs
            # info kann jetzt "stale" positions enthalten -> wir überschreiben eh gleich via _get_xy_from_info()
            info.pop("x_position", None)
            info.pop("y_position", None)

        if self.goal_queue:
            self.goal = np.array(self.goal_queue.pop(0), dtype=np.float64)
            self.is_random_wandering = False
            self.logger.info(f"🔄 Reset: Popped first goal from series: {self.goal}")
        elif target_goal is not None:
            self.goal = np.array(target_goal, dtype=np.float64)
            self.is_random_wandering = False
            self.logger.info(f"🔄 Reset: Set specific target goal: {self.goal}")
        else:
            self.goal = self._sample_goal()
            # self.logger.info(f"🔄 Reset: No goal provided. Sampling Random: {self.goal}") # Spams training loop

        xy_pos = self._get_xy_from_info(info)
        self.prev_xy_pos = xy_pos.copy()

        hint, _, dist = self._get_obs_components(obs, xy_pos)
        obs = np.concatenate([obs, hint])

        info["goal_x"] = self.goal[0]
        info["goal_y"] = self.goal[1]
        info["dist_to_goal"] = dist
        return obs, info

    def step(self, action):

        prev_xy = self.prev_xy_pos.copy()
        
        obs_raw, rewards, terminated, truncated, info = self.env.step(action)
        
        xy_pos = self._get_xy_from_info(info)
        self.prev_xy_pos = xy_pos.copy()

        # --- Compute Components ---
        hint, current_tilt, dist_now = self._get_obs_components(obs_raw, xy_pos)
        
        # --- Reward Logic ---
        reward_forward = info.get("reward_forward", info.get("forward_reward", 0.0))
        others = rewards - reward_forward
        scaled_others = others * self.current_healthy_scale

        dt = getattr(self.env.unwrapped, "dt", 0.05)
        # 🟢 FAILSAFE: Division by zero check
        if dt <= 1e-8:
             self.logger.warning(f"⚠️ dt is extremely small ({dt}). Using 1e-8 to avoid NaN velocity.")
             dt = 1e-8

        vel_xy = (xy_pos - prev_xy) / dt
        
        vec_to_goal = self.goal - prev_xy
        dist_prev = np.linalg.norm(vec_to_goal)
        
        v_towards = 0.0
        # 🟢 FAILSAFE: Don't normalize zero vectors
        if dist_prev > 1e-8:
            u_dir = vec_to_goal / dist_prev
            v_towards = np.dot(vel_xy, u_dir)
        else:
            # We are ON the goal, velocity towards it is effectively zero/undefined
            v_towards = 0.0 
        if self.vel_rew_max_speed > 0:
            v_towards = np.clip(v_towards, -self.vel_rew_max_speed, self.vel_rew_max_speed)
            
        vel_rew = self.vel_scale * v_towards
        if vel_rew < 0:
            vel_rew *= self.scale_wrong_direction

        excess_tilt = np.maximum(0.0, current_tilt - self.tilt_deadzone)
        tilt_pen = -1.0 * excess_tilt * self.current_tilt_pen
        
        speed_scalar = np.linalg.norm(vel_xy)
        move_bonus = np.minimum(speed_scalar, 1.0) * self.current_move_bonus

        reached = dist_now < self.success_radius
        success_bonus = 0.0
        
        if reached:
            success_bonus = 10.0
            if self.goal_queue:
                self.goal = np.array(self.goal_queue.pop(0), dtype=np.float64)
                hint, _, dist_now = self._get_obs_components(obs_raw, xy_pos)
                self.is_random_wandering = False
                self.logger.info(f"🎉 Goal Reached! Switched to next in queue: {self.goal}")
            else:
                if not hasattr(self, 'keep_goal') or not self.keep_goal:
                     self.goal = self._sample_goal()
                     hint, _, dist_now = self._get_obs_components(obs_raw, xy_pos)
                     self.logger.info(f"🎉 Goal Reached! Queue empty. Resampled Random: {self.goal}")

        obs = np.concatenate([obs_raw, hint])

        death_penalty = self.death_penalty if (terminated and not truncated) else 0.0

        total_reward = scaled_others + vel_rew + tilt_pen + move_bonus + success_bonus + death_penalty

        info["goal_x"] = self.goal[0]
        info["goal_y"] = self.goal[1]
        info["dist_to_goal"] = dist_now
        info["velocity_reward"] = vel_rew
        info["tilt_penalty"] = tilt_pen
        info["success_bonus"] = success_bonus
        info["reached_goal"] = reached

        return obs, total_reward, terminated, truncated, info

    def update_angle_range(self, max_rad: float):
        """Optional: von außen die Yaw-Range setzen (z.B. Curriculum)."""
        self.current_max_yaw = float(max_rad)

    def _apply_random_spawn_pose(self, seed=None):
        """Modifies MuJoCo state: joint noise + yaw randomization + vel noise."""
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        base = self.env.unwrapped
        if not (hasattr(base, "model") and hasattr(base, "init_qpos") and hasattr(base, "init_qvel")):
            # Kein Mujoco/Ant -> nichts tun
            return None

        model = base.model
        qpos = base.init_qpos.copy()
        qvel = base.init_qvel.copy()

        # Joint noise (ab Index 7)
        if model.nq > 7 and self.spawn_noise_scale > 0:
            joint_noise = self._rng.uniform(-self.spawn_noise_scale, self.spawn_noise_scale, size=model.nq - 7)
            qpos[7:] += joint_noise

        # Yaw randomization: q = [cos(a/2), 0, 0, sin(a/2)]
        if self.randomize_yaw and self.current_max_yaw > 0:
            a = self._rng.uniform(-self.current_max_yaw, self.current_max_yaw)
            yaw_quat = np.array([np.cos(a / 2), 0.0, 0.0, np.sin(a / 2)], dtype=np.float64)
            qpos[3:7] = yaw_quat
        else:
            # optional: tiny quat noise to break symmetry
            if self.spawn_noise_scale > 0:
                quat_noise = self._rng.uniform(-self.spawn_noise_scale, self.spawn_noise_scale, size=4)
                qpos[3:7] = qpos[3:7] + quat_noise
                qpos[3:7] /= (np.linalg.norm(qpos[3:7]) + 1e-8)

        # Height
        qpos[2] = self.spawn_height

        # Velocity noise
        if self.spawn_vel_noise_scale > 0:
            qvel = qvel + self._rng.normal(0.0, self.spawn_vel_noise_scale, size=model.nv)

        # Commit
        base.set_state(qpos, qvel)

        # Settle physics
        for _ in range(self.spawn_settle_steps):
            base.do_simulation(np.zeros(model.nu), base.frame_skip)

        # Fresh raw obs from base env (bypasses inner wrappers!)
        return base._get_obs()
