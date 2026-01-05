import torch
import os
import gymnasium as gym

def limit_threads(n: int):
    # PyTorch threads
    torch.set_num_threads(n)
    torch.set_num_interop_threads(1)

    # NumPy / BLAS threads (muss vor Imports passieren, aber auch so meist ok)
    os.environ["OMP_NUM_THREADS"] = str(n)
    os.environ["OPENBLAS_NUM_THREADS"] = str(n)
    os.environ["MKL_NUM_THREADS"] = str(n)
    os.environ["NUMEXPR_NUM_THREADS"] = str(n)


def make_train_env(env_id, seed):
    
    env = gym.make(env_id)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = gym.wrappers.ClipAction(env)
    return env

def make_eval_env(env_id, seed, capture_video, run_name, name_prefix="rollout"):
    seed_offset = seed + 1000
    if capture_video:
        env = gym.make(env_id, render_mode="rgb_array")

        # in dieser (frisch erzeugten) video-env: genau Episode 0 aufnehmen
        env = gym.wrappers.RecordVideo(
            env,
            f"videos/{run_name}",
            name_prefix=name_prefix,
            episode_trigger=lambda ep: ep == 0,
        )
    else:
        env = gym.make(env_id)

    env.action_space.seed(seed_offset)
    env.observation_space.seed(seed_offset)

    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = gym.wrappers.ClipAction(env)
    return env

def make_video_env(args, run_name, name_prefix: str):
        return make_eval_env(args.env_id,args.seed, True, run_name, name_prefix)