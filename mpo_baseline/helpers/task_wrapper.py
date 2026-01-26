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
        
        # 🟢 NEW: Curriculum Speed Limit
        self.current_max_speed = float('inf') 

        # 🟢 CRITICAL FIX FOR HALF-CHEETAH / ANT:
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
        
        # 🟢 Inject Hint
        obs = self._add_hint(obs)
        
        # Add telemetry
        info['task_direction'] = self.invert
        return obs, info
    
    def step(self, action):
        obs, rewards, terminated, truncated, info = self.env.step(action)
        
        # 🟢 Inject Hint
        obs = self._add_hint(obs)        

        reward_forward = info.get("reward_forward")
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
