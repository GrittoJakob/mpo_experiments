import gymnasium as gym
from .task_wrapper import GoalPositionWrapper
from .meta_task_wrapper import Meta_InvertedWrapper 
from .ERFI_Wrappers import RAOActionWrapper, RFIActionWrapper

import numpy as np

def make_ant_env(env_id, args, render_mode: None):

    # For custom designed reward function
    kwargs = dict(
        ctrl_cost_weight=args.ctrl_cost_weight,
        healthy_reward=args.healthy_reward_weight,
        contact_cost_weight=args.contact_cost_weight,
        forward_reward_weight=args.forward_reward_weight,
        include_cfrc_ext_in_observation =args.include_cfrc_ext_in_observation,
    )

    if render_mode is not None:
        kwargs["render_mode"] = render_mode

    env = gym.make(env_id, **kwargs)

    # Wrap observations and custom designed reward function for tasks
    env = wrap_task_for_robust_ant(env, args)

    return env

def wrap_task_for_robust_ant(env, args):
    """
    Apply task and randomization wrappers.

    task_mode:
        - "default"
        - "target_goal"
        - "inverted_without_task_hint"

    rand_mode:
        - "default"
        - "RFI"
        - "RAO"
    """
    task_mode = getattr(args, "task_mode", "default")
    rand_mode = getattr(args, "rand_mode", "default")
    noise_limit = getattr(args, "noise_limit", 0.05)

  
    # Task wrappers
    if task_mode == "target_goal":
        env = GoalPositionWrapper(env, args)

    elif task_mode == "inverted_without_task_hint":
        env = Meta_InvertedWrapper(
            env,
            args,
            args.history_len,
            args.append_task_reward,
        )

    # Randomization wrappers
    if rand_mode == "RFI":
        print("RFI is activated")
        env = RFIActionWrapper(env, noise_limit)

    elif rand_mode == "RAO":
        print("RAO is activated")
        env = RAOActionWrapper(env, noise_limit)

    return env
