import torch
from typing import Optional
import os
import gymnasium as gym
from .task_wrapper import InvertedVelocityWrapper, GoalPositionWrapper
from .multi_task_wrapper import Multi_Task_InvertedWrapper
from .ERFI_Wrappers import RFIActionWrapper

def limit_threads(n: int):
    # PyTorch threads
    torch.set_num_threads(n)
    torch.set_num_interop_threads(1)

    # NumPy / BLAS threads (muss vor Imports passieren, aber auch so meist ok)
    os.environ["OMP_NUM_THREADS"] = str(n)
    os.environ["OPENBLAS_NUM_THREADS"] = str(n)
    os.environ["MKL_NUM_THREADS"] = str(n)
    os.environ["NUMEXPR_NUM_THREADS"] = str(n)

def maybe_wrap_task(env, args):
    if getattr(args, "task_mode", "default") == "inverted":
        env = InvertedVelocityWrapper(env, args)
    if getattr(args, "task_mode", "default") == "target_goal":
        env = GoalPositionWrapper(env, args)
    if getattr(args, "task_mode", "default") == "inverted_multi_task":
        env = Multi_Task_InvertedWrapper(env, args, args.history_len, args.append_task_reward)
    if args.rand_mode == "RFI":
        # Apply RFI to ALL environments
        print("RFI is activated")
        env = RFIActionWrapper(env, args.final_rand_noise)
    # elif args.rand_mode == "RAO":
    #     # Apply RAO to ALL environments
    #     env = mw.RAOActionWrapper(env, args.noise_limit)

    # elif args.rand_mode == "ERFI":
    #     # Split the population based on rank and ratio
    #     if rank < split_idx:
    #         env = mw.RFIActionWrapper(env, args.noise_limit)
    #     else:
    #         env = mw.RAOActionWrapper(env, args.noise_limit)

    return env


def _make_base_env(env_id: str, args, render_mode: Optional[str] = None):
    """Central Builder, for equality"""
    if env_id == "Ant-v5":
        kwargs = dict(
            ctrl_cost_weight=args.ctrl_cost_weight,
            healthy_reward=args.healthy_reward_weight,
            contact_cost_weight=args.contact_cost_weight,
            forward_reward_weight=args.forward_reward_weight,
            include_cfrc_ext_in_observation=args.include_cfrc_ext_in_observation,
        )
        if render_mode is not None:
            kwargs["render_mode"] = render_mode
        env = gym.make(env_id, **kwargs)
    else:
        if render_mode is None:
            env = gym.make(env_id)
        else:
            env = gym.make(env_id, render_mode=render_mode)
    return env


def make_train_env(args, env_id, seed):
    env = _make_base_env(env_id, args, render_mode=None)

    env = maybe_wrap_task(env, args)

    env.action_space.seed(seed)
    env.observation_space.seed(seed)

    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = gym.wrappers.ClipAction(env)
    return env


def make_eval_env(args, env_id, seed, capture_video, run_name, name_prefix="rollout"):
    seed_offset = seed + 1000

    if capture_video:
        env = _make_base_env(env_id, args, render_mode="rgb_array")
        env = gym.wrappers.RecordVideo(
            env,
            f"videos/{run_name}",
            name_prefix=name_prefix,
            episode_trigger=lambda ep: ep == 0,
        )
    else:
        env = _make_base_env(env_id, args, render_mode=None)

    env = maybe_wrap_task(env, args)

    env.action_space.seed(seed_offset)
    env.observation_space.seed(seed_offset)

    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = gym.wrappers.ClipAction(env)
    return env


def make_video_env(args, run_name, name_prefix: str):
    return make_eval_env(args, args.env_id, args.seed, True, run_name, name_prefix)

def _train_env_thunk(args, env_id: str, base_seed: int, rank: int, threads_per_worker: int):
    
    def _thunk():
        # In Subprozessen Thread-Anzahl klein halten, sonst wird's langsamer
        # if threads_per_worker is not None:
        #     limit_threads(int(threads_per_worker))

        seed = int(base_seed) + int(rank)
        return make_train_env(args, env_id, seed)
    return _thunk


def make_train_vec_env(
    args,
    env_id: str,
    seed: int,
    num_envs: int,
    asynchronous: bool = True,
    threads_per_worker: int = 1,
    mp_context=None,
):
    assert num_envs >= 1

    apply_thread_limits = (asynchronous and num_envs > 1)

    env_fns = [
        _train_env_thunk(
            args, env_id, seed,
            rank=i,
            threads_per_worker=(threads_per_worker if apply_thread_limits else None)
        )
        for i in range(num_envs)
    ]

    if asynchronous and num_envs > 1:
        return gym.vector.AsyncVectorEnv(env_fns, context=mp_context)
    else:
        return gym.vector.SyncVectorEnv(env_fns)