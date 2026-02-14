import gymnasium as gym
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
from typing import Optional
import logging

class ExcludeFloorForceWrapper(gym.ObservationWrapper):
    """
    Surgically removes the 6 contact force values corresponding to 'floor_body'.
    """
    def __init__(self, env, log_level=logging.ERROR):
        super().__init__(env)

        # --- 1. Logger Setup ---
        self.logger = logging.getLogger("ExcludeFloorForceWrapper")
        # Prevent adding multiple handlers if wrapper is instantiated multiple times
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('[ExcludeFloorForce] %(levelname)s: %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        self.logger.setLevel(log_level)
        
        # --- 2. Initialization ---
        self.floor_body_id = -1
        self.start_idx = 0
        self.cfrc_len = 0
        
        try:
            if hasattr(env.unwrapped, "model"):
                model = env.unwrapped.model
                
                # A. Find Floor Body ID
                self.floor_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "floor_body")
                
                # B. Calculate Observation Shapes
                self.n_bodies = model.nbody
                self.cfrc_len = self.n_bodies * 6
                total_obs_len = env.observation_space.shape[0]
                
                # Contact forces are always at the end of the Ant-v5 observation vector
                self.start_idx = total_obs_len - self.cfrc_len
                
                # C. Validation
                if self.start_idx < 0:
                    self.logger.error("Observation too small for calculated contact forces. Disabling wrapper.")
                    self.floor_body_id = -1
                    
        except Exception as e:
            self.logger.error(f"Init Error: {e}")
            self.floor_body_id = -1

        # --- 3. Update Observation Space ---
        if self.floor_body_id != -1:
            # We are removing exactly 6 numbers (1 body * 6 forces)
            current_len = env.observation_space.shape[0]
            new_len = current_len - 6
            
            # Create new Box space
            low = -np.inf * np.ones(new_len, dtype=np.float64)
            high = np.inf * np.ones(new_len, dtype=np.float64)
            self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float64)
            
            # Log success
            start_cut = self.floor_body_id * 6
            end_cut = start_cut + 6
            self.logger.info(f"✂️ Removing sensors for body {self.floor_body_id} (Indices {start_cut}-{end_cut} relative to CFRC start)")
            
        else:
            self.logger.warning("⚠️ 'floor_body' not found. Wrapper will pass observations through unchanged.")

    def observation(self, obs):
        # Case 1: Wrapper disabled (floor not found)
        if self.floor_body_id == -1:
            return obs

        # Case 2: Safety Check (Input size already matches output size)
        # This happens if applied to a Training Env that lacks the XML patch (obs=108)
        if obs.shape[0] == self.observation_space.shape[0]:
             return obs

        # Case 3: Surgical Removal
        # 1. Split Obs into [State, ContactForces]
        state_part = obs[:self.start_idx]
        cfrc_part = obs[self.start_idx:]
        
        # 2. Calculate Cut indices within the cfrc_part
        cut_start = self.floor_body_id * 6
        cut_end = cut_start + 6
        
        # 3. Slice out the floor forces
        new_cfrc = np.concatenate([cfrc_part[:cut_start], cfrc_part[cut_end:]])
        
        return np.concatenate([state_part, new_cfrc])

class PhysicsDynamicsWrapper(gym.Wrapper):
    """
    Stateful Physics Wrapper.
    - Real-time updates via sliders.
    - Persistent settings across resets.
    - Correct 3D rotation math (Tilt then Spin).
    """
    def __init__(self, env: gym.Env, log_level=logging.ERROR):
        super().__init__(env)
        
        # --- Internal State ---
        self.friction_mult = 1.0
        self.slope_deg = 0.0
        self.slope_yaw = 0.0
        self.floor_id = -1
        
        # Persistence Memory
        self.fixed_slope_deg = None 
        self.fixed_slope_yaw = None

        # Survival bounds
        self.min_healthy_z = 0.05
        self.max_healthy_z = 2
        
        # Cache
        self.floor_rot = Rotation.identity()

        # Logger
        self.logger = logging.getLogger("PhysicsDynamics")
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('[PhysicsWrapper] %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        self.logger.setLevel(log_level)

        if hasattr(env.unwrapped, "terminate_when_unhealthy"):
             env.unwrapped._terminate_when_unhealthy = False 
        
        # Randomization Limits
        self.min_slope_limit = 0.0
        self.max_slope_limit = 0.0 

    def set_slope_range(self, min_deg, max_deg):
        self.min_slope_limit = float(min_deg)
        self.max_slope_limit = float(max_deg)

    # =========================================================================
    # SLIDER INTERFACE (Updates Memory AND Current State)
    # =========================================================================
    def set_slope(self, angle_degrees: float):
        val = float(angle_degrees)
        # 1. Update Memory (for next reset)
        self.fixed_slope_deg = val
        # 2. Update Current State (for immediate physics)
        self.slope_deg = val
        
        self._apply_physics()
        self.logger.info(f"📐 Slope set to {self.slope_deg}°")

    def set_slope_direction(self, yaw_degrees: float):
        val = float(yaw_degrees)
        # 1. Update Memory
        self.fixed_slope_yaw = val
        # 2. Update Current State
        self.slope_yaw = val
        
        self._apply_physics()
        self.logger.info(f"🧭 Direction set to {self.slope_yaw}°")

    def set_friction(self, friction_mult: float):
        self.friction_mult = float(friction_mult)
        self._apply_physics() 

    # =========================================================================
    # CORE LOGIC
    # =========================================================================
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        # 1. Resolve Slope Steepness
        if options and "slope_deg" in options:
            self.slope_deg = float(options["slope_deg"])
        elif self.fixed_slope_deg is not None:
            self.slope_deg = self.fixed_slope_deg
        else:
            self.slope_deg = self.np_random.uniform(self.min_slope_limit, self.max_slope_limit)
            if self.np_random.random() < 0.5: self.slope_deg *= -1

        # 2. Resolve Slope Direction
        if options and "slope_yaw" in options:
            self.slope_yaw = float(options["slope_yaw"])
        elif self.fixed_slope_yaw is not None:
            self.slope_yaw = self.fixed_slope_yaw
        else:
            if self.max_slope_limit > 0:
                self.slope_yaw = self.np_random.uniform(-180, 180)
            else:
                self.slope_yaw = 0.0

        # 3. Apply
        self._apply_physics()
        
        # 4. Standard Reset
        obs, info = super().reset(seed=seed, options=options)
        
        # 5. Fix Obs
        obs = self._correct_observation(obs)
        
        info["slope_deg"] = self.slope_deg
        info["slope_yaw"] = self.slope_yaw
        
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        obs = self._correct_observation(obs)
        
        relative_z = obs[0]
        is_healthy = self.min_healthy_z <= relative_z <= self.max_healthy_z
        
        # If UNHEALTHY relative to floor -> Kill it.
        if not is_healthy:
            terminated = True
            
        # If HEALTHY relative to floor -> Keep it alive (Override Base Env)
        else:
            terminated = False
        
        if "x_position" not in info:
             try:
                 qpos = self.env.unwrapped.data.qpos
                 info["x_position"] = qpos[0]
                 info["y_position"] = qpos[1]
             except: pass

        info["relative_height"] = relative_z
        return obs, reward, terminated, truncated, info

    def _apply_physics(self):
        """Applies physics to MuJoCo."""
        if hasattr(self.env.unwrapped, "model"):
            model = self.env.unwrapped.model
            model.geom_friction[:] = self.friction_mult
            
            try:
                if self.floor_id == -1:
                    self.floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "floor_body")
                
                if self.floor_id != -1:
                    # 🟢 EXPLICIT COMPOSITION: R_final = R_yaw * R_tilt
                    # 1. Create Tilt (Around Y, creating an X-slope)
                    #    -slope_deg means positive X goes UP.
                    r_tilt = Rotation.from_euler('y', -self.slope_deg, degrees=True)
                    
                    # 2. Create Yaw (Around Z)
                    r_yaw = Rotation.from_euler('z', self.slope_yaw, degrees=True)
                    
                    # 3. Compose: Yaw * Tilt
                    # This rotates the Tilted Plane to face the Yaw direction.
                    self.floor_rot = r_yaw * r_tilt
                    
                    scipy_quat = self.floor_rot.as_quat() # x, y, z, w
                    mujoco_quat = np.array([scipy_quat[3], scipy_quat[0], scipy_quat[1], scipy_quat[2]])
                    
                    model.body_quat[self.floor_id] = mujoco_quat
                    model.opt.gravity[:] = [0, 0, -9.81]
            except Exception as e:
                self.logger.error(f"Physics Error: {e}")

    def _correct_observation(self, obs):
        global_pos = self.env.unwrapped.data.qpos[:3].copy()
        r_inv = self.floor_rot.inv()
        local_pos = r_inv.apply(global_pos)
        obs[0] = local_pos[2]
        return obs
