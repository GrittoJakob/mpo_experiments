import os
os.environ["OMP_NUM_THREADS"] = str(4)
os.environ["OPENBLAS_NUM_THREADS"] = str(4)
os.environ["MKL_NUM_THREADS"] = str(4)
os.environ["NUMEXPR_NUM_THREADS"] = str(4)
os.environ["MAX_JOBS"] = "16" #for torch compile
os.environ["TORCH_LOGS"] = "+recompiles, +graph_breaks"
os.environ["TORCH_COMPILE_DEBUG"] = "1"

import time
import random
from dataclasses import dataclass
import numpy as np
import torch
import tyro
import copy
from torch.utils.tensorboard import SummaryWriter
import gymnasium as gym
import wandb
from nets.actor import Actor
from nets.critic import Critic
from typing import Optional
from environment.env_creator import limit_threads, make_train_env, make_eval_env, make_train_vec_env
from buffer.replaybuffer import ReplayBuffer
from buffer.replaybuffer_gpu import ReplayBufferGPU
from train_args import Args
from train_mpo.train_loop import train_loop
from helpers.warm_up_compilation import warmup_mpo_compile

from mpo.__init__ import MPO

def make_envs(args, run_name):
    if args.num_envs > 1:
        train_env = make_train_vec_env(
            args, 
            args.env_id,
            args.seed,
            args.num_envs,
            args.asynchronous,
            args.num_threads
        )
    else:
        train_env = make_train_env(args, args.env_id, args.seed)
    
    eval_env = make_eval_env(args, args.env_id, args.seed, capture_video = False, run_name = run_name, name_prefix = "eval")
    if isinstance(train_env, gym.vector.VectorEnv):
        args.obs_space   = int(np.prod(train_env.single_observation_space.shape))
        args.action_dim  = int(np.prod(train_env.single_action_space.shape))
    else:
        args.obs_space   = int(np.prod(train_env.observation_space.shape))
        args.action_dim  = int(np.prod(train_env.action_space.shape))
    return train_env, eval_env

def init_runname(args):
     # Run-Name à la CleanRL
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    return run_name
    
def load_actor_critic_checkpoint(
    checkpoint_path: str,
    actor: torch.nn.Module,
    critic: torch.nn.Module,
    target_actor: Optional[torch.nn.Module],
    target_critic: Optional[torch.nn.Module],
    device: torch.device,
    strict: bool = True,
    ):
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location=device,weights_only=False)

    actor.load_state_dict(ckpt["actor_state_dict"], strict=strict)
    critic.load_state_dict(ckpt["critic_state_dict"], strict=strict)

    # Targets direkt auf denselben Stand bringen
    if target_actor is not None:
        target_actor.load_state_dict(actor.state_dict(), strict=True)
    if target_critic is not None:
        target_critic.load_state_dict(critic.state_dict(), strict=True)

    print(
        f"[LOAD] Loaded actor+critic from: {checkpoint_path} | "
        f"steps={ckpt.get('num_steps')} gu={ckpt.get('grad_updates')}"
    )
    return ckpt


def make_networks(args, device):
    
    actor = Actor(args).to(device)
    critic = Critic(args).to(device)
    # Target networks (used for stable targets)
    target_actor = copy.deepcopy(actor).to(device)
    target_critic = copy.deepcopy(critic).to(device)
    return actor, critic, target_actor, target_critic
     
def make_optimizer(args, actor, critic):
    actor_optimizer = torch.optim.Adam(actor.parameters(), args.actor_lr)
    critic_optimizer = torch.optim.Adam(critic.parameters(), args.critic_lr)
    return actor_optimizer, critic_optimizer

def train():
    args = tyro.cli(Args)
    limit_threads(args.num_threads)
    
    # Device
    device = torch.device("cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")
    print("Device: ", device)


    # Run-Name 
    run_name = init_runname(args)
    args.run_name = run_name
    
    # Logging-Verzeichnis
    os.makedirs(args.log_dir, exist_ok=True)
    tb_dir = os.path.join(args.log_dir, "tb", run_name)

    # Create train_env and eval_env
    train_env, eval_env = make_envs(args, args.run_name)
    assert isinstance(train_env.action_space, gym.spaces.Box)
    
    # Seeding 
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False    
    train_env.reset(seed=args.seed)  

    # Create Networks
    actor, critic, target_actor, target_critic = make_networks(args, device)

    #Load nets
    
    _ckpt = load_actor_critic_checkpoint(
        checkpoint_path=args.ckpt_path,
        actor=actor,
        critic=critic,
        target_actor=target_actor,
        target_critic=target_critic,
        device=device,
        strict=True,
    )


    # Create Optimizer
    actor_optimizer, critic_optimizer = make_optimizer(args, actor, critic)

    # Create ReplayBuffer
    if args.buffer_on_cuda and (args.device == "cuda" and torch.cuda.is_available()):
        replaybuffer = ReplayBufferGPU(args.max_replay_buffer, device = "cuda")
        gpu_buffer = True
    else:
        replaybuffer = ReplayBuffer(args.max_replay_buffer)
        gpu_buffer = False

    # action space
    action_space = eval_env.unwrapped.action_space
    args.action_space_low  = action_space.low
    args.action_space_high = action_space.high
    print("action space:", args.action_space_low, args.action_space_high)
    mpo = MPO(args, eval_env, actor, target_actor, critic, target_critic, actor_optimizer, critic_optimizer, device) 
    
    #  compile warmup 
    if getattr(args, "use_compile", True):
        mpo = warmup_mpo_compile(
            args=args,
            device=device,
            env=train_env,          
            mpo=mpo,
            compile_mode=getattr(args, "compile_mode", "reduce-overhead"),
        )  

    train_loop(args, train_env, eval_env, device, replaybuffer, mpo, gpu_buffer)


if __name__ == "__main__":
    train()

  


