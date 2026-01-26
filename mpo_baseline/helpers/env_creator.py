import torch
import os
import gymnasium as gym
from .task_wrapper import InvertedVelocityWrapper


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
    return env



def make_train_env(args, env_id, seed):
    if env_id == "Ant-v5":
        env = gym.make(
            env_id, 
            ctrl_cost_weight = args.ctrl_cost_weight,
            healthy_reward = args.healthy_reward_weight, 
            contact_cost_weight = args.contact_cost_weight,
            forward_reward_weight = args.forward_reward_weight
            )

    else: 
        env = gym.make(env_id)

    env = maybe_wrap_task(env, args)

    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = gym.wrappers.ClipAction(env)
    return env

def make_eval_env(args, env_id, seed, capture_video, run_name, name_prefix="rollout"):
    seed_offset = seed + 1000
    if capture_video:
        env = gym.make(
            env_id, 
            render_mode="rgb_array",
            ctrl_cost_weight = args.ctrl_cost_weight,
            healthy_reward = args.healthy_reward_weight, 
            contact_cost_weight = args.contact_cost_weight,
            forward_reward_weight = args.forward_reward_weight
            )

        # in dieser (frisch erzeugten) video-env: genau Episode 0 aufnehmen
        env = gym.wrappers.RecordVideo(
            env,
            f"videos/{run_name}",
            name_prefix=name_prefix,
            episode_trigger=lambda ep: ep == 0,
        )
    else:
        env = gym.make(
            env_id, 
            ctrl_cost_weight = args.ctrl_cost_weight,
            healthy_reward = args.healthy_reward_weight, 
            contact_cost_weight = args.contact_cost_weight,
            forward_reward_weight = args.forward_reward_weight
            )

    env = maybe_wrap_task(env, args)
    env.action_space.seed(seed_offset)
    env.observation_space.seed(seed_offset)

    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = gym.wrappers.ClipAction(env)
    return env

def make_video_env(args, run_name, name_prefix: str):
        return make_eval_env(args, args.env_id,args.seed, True, run_name, name_prefix)