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
# 2. RAO Wrapper (Episodic Bias) - UPDATED
# ==============================================================================
class RAOActionWrapper(gym.ActionWrapper):
    def __init__(self, env: gym.Env, noise_limit: float = 0.05):
        super().__init__(env)
        self.default_limit = noise_limit
        self.current_limit = noise_limit
        self.bias = np.zeros(env.action_space.shape, dtype=env.action_space.dtype)
    
    def set_noise_limit(self, new_limit: float):
        """Directly set the noise limit (bypassing curriculum)."""
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



#  in env_creator

# !!! before loop 
# # Calculate the split index based on the ratio
#     # e.g., if num_envs=10 and ratio=0.8 -> split_idx=8. 
#     # Envs 0-7 get RFI, Envs 8-9 get RAO.
#     split_idx = int(args.num_envs * getattr(args, "rand_split_ratio", 0.5))

# !!! in loop before all other wrappers: noise_limit can be set directly, but increasing it gradually with curriculum probably safer
#             if args.rand_mode == "RFI":
#                 # Apply RFI to ALL environments
#                 env = mw.RFIActionWrapper(env, noise_limit=0.0)

#             elif args.rand_mode == "RAO":
#                 # Apply RAO to ALL environments
#                 env = mw.RAOActionWrapper(env, noise_limit=0.0)

#             elif args.rand_mode == "ERFI":
#                 # Split the population based on rank and ratio
#                 if rank < split_idx:
#                     env = mw.RFIActionWrapper(env, noise_limit=0.0)
#                 else:
#                     env = mw.RAOActionWrapper(env, noise_limit=0.0)

# !!! curriculum during training, set inside training loop before all other logic
#   if args.rand_mode is not None:
#                     # Update Rand. Perturbation Magnitude
#                     current_noise_limit = args.start_rand_noise + progress * (args.final_rand_noise - args.start_rand_noise)
# #                     envs.call("set_noise_limit", current_noise_limit)
# #                     writer.add_scalar("curriculum/rand_noise_limit", current_noise_limit, global_step)

# !!! in train_args.py

#     rand_mode: str = "ERFI"
#     """the environment parametrization type for meta-learning""" # Currently supports ERFI, RAO, RFI, and None (no param randomization)
#     rand_split_ratio: float = 0.5
#     """the ratio at which to split the population for ERFI mode, if 0.9 the first 90% of the population uses RFI and the last 10% uses RAO"""
#     start_rand_noise: float = 0.0
#     """the initial randomization scale for the environment parameters sampling"""
#     final_rand_noise: float = 0.05
#     """the final randomization scale for the environment parameters sampling"""            


# !!! in the eval to set noise deterministically, could also be split into just RAO and RFI

#  options = {}
# curr_rand_noise = rand_noise_limits[r]
# options["rfi_limit"] = curr_rand_noise
# options["rao_limit"] = curr_rand_noise
# obs, _ = current_env.reset(seed=args.seed + 10000 + r, options=options)


