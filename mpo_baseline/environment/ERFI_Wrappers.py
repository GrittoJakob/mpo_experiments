import gymnasium as gym
import numpy as np
from typing import Optional

# Task Wrappers from Ferdinand -> Thanks!
# ==============================================================================
# 1. RFI Wrapper (Step-wise Noise) - UPDATED
# ==============================================================================
class RFIActionWrapper(gym.ActionWrapper):
    def __init__(self, env: gym.Env, noise_limit: float = 0.05):
        super().__init__(env)
        self.default_limit = noise_limit
        self.current_limit = noise_limit
        
        # Validation
        assert isinstance(env.action_space, gym.spaces.Box), "RFI requires Box action space"

    def set_noise_limit(self, new_limit: float):
        """Directly set the noise limit (bypassing curriculum)."""
        self.current_limit = float(new_limit)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        # 1. Check for overrides in options
        if options and "rfi_limit" in options:
            self.current_limit = float(options["rfi_limit"])
        else:
            self.current_limit = self.default_limit
            
        return super().reset(seed=seed, options=options)

    def action(self, action: np.ndarray) -> np.ndarray:
        # Use self.current_limit instead of fixed limit
        if self.current_limit <= 0.0:
            return action
            
        noise = self.np_random.uniform(
            low=-self.current_limit,
            high=self.current_limit,
            size=action.shape
        )
        
        perturbed_action = action + noise
        low = self.action_space.low
        high = self.action_space.high
        return np.clip(perturbed_action, low, high)

# ==============================================================================
# 2. RAO Wrapper (Episodic Bias) 
# ==============================================================================
class RAOActionWrapper(gym.ActionWrapper):
    def __init__(self, env: gym.Env, noise_limit: float = 0.05):
        super().__init__(env)
        self.default_limit = noise_limit
        self.current_limit = noise_limit
        self.bias = np.zeros(env.action_space.shape, dtype=env.action_space.dtype)
    
    def set_noise_limit(self, new_limit: float):
        """Directly set the noise limit"""
        self.current_limit = float(new_limit)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        # 1. Check for overrides in options
        if options and "rao_limit" in options:
            self.current_limit = float(options["rao_limit"])
        else:
            self.current_limit = self.default_limit
        
        # 2. Sample Bias using the (potentially new) limit
        # Note: We must call super().reset() to ensure seeds are set if needed
        obs, info = super().reset(seed=seed, options=options)
        
        if self.current_limit > 0.0:
            self.bias = self.np_random.uniform(
                low=-self.current_limit,
                high=self.current_limit,
                size=self.action_space.shape
            )
        else:
            self.bias = np.zeros_like(self.bias)
            
        info["rao_bias"] = self.bias
        return obs, info

    def action(self, action: np.ndarray) -> np.ndarray:
        perturbed_action = action + self.bias
        low = self.action_space.low
        high = self.action_space.high
        return np.clip(perturbed_action, low, high)


class ERFIEvalActionWrapper(gym.ActionWrapper):
    """
    Evaluation wrapper: on reset() choose mode via options["erfi_mode"] in {"RFI","RAO"}.
    Then action() applies the corresponding perturbation.
    """
    def __init__(self, env, *, rfi_noise, rao_noise, rng_seed: int = 0):
        super().__init__(env)
        self.rfi_noise = float(rfi_noise)
        self.rao_noise = float(rao_noise)
        self.rng = np.random.default_rng(rng_seed)
        self.mode = "RFI"  # default

    def reset(self, *, seed=None, options=None):
        options = {} if options is None else dict(options)

        # custom option rausziehen, damit base env nicht meckert
        mode = options.pop("erfi_mode", None)
        if mode is not None:
            if mode not in ("RFI", "RAO"):
                raise ValueError(f"erfi_mode must be 'RFI' or 'RAO', got {mode}")
            self.mode = mode

        obs, info = self.env.reset(seed=seed, options=options)
        info = dict(info) if isinstance(info, dict) else {}
        info["erfi_mode"] = self.mode
        return obs, info

    def action(self, action):
        # hier exakt deine RFI/RAO-Logik rein (oder aus ausgelagerten funcs aufrufen)
        if self.mode == "RFI":
            # Beispiel: additiver Gaussian noise
            noise = self.rfi_noise * self.rng.standard_normal(size=np.shape(action))
            return action + noise
        else:
            # Beispiel: clipped noise (RAO-artig)
            noise = self.rao_noise * self.rng.standard_normal(size=np.shape(action))
            noise = np.clip(noise, -self.rao_noise, self.rao_noise)
            return action + noise
