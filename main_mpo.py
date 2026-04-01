import os
_DEFAULT_THREADS = "4"
os.environ["OMP_NUM_THREADS"] = _DEFAULT_THREADS
os.environ["OPENBLAS_NUM_THREADS"] = _DEFAULT_THREADS
os.environ["MKL_NUM_THREADS"] = _DEFAULT_THREADS
os.environ["NUMEXPR_NUM_THREADS"] = _DEFAULT_THREADS
os.environ["MAX_JOBS"] = "16"
os.environ["TORCH_LOGS"] = "+recompiles,+graph_breaks"
os.environ["TORCH_COMPILE_DEBUG"] = "1"

import time
import random
from dataclasses import dataclass
import numpy as np
import torch
import tyro
import copy
import gymnasium as gym
from typing import Union, Annotated
from nets.MLP_actor import Actor
from nets.MLP_critic import Critic
from environment.base_env_creator import limit_threads, make_eval_env, make_train_vec_env, make_base_env
from buffer.single_step_replaybuffer import ReplayBuffer
from buffer.episodic_replaybuffer import EpisodicReplayBuffer
from configs.Robust_Ant_v5 import Robust_Ant_Args
from configs.Ant_Maze import Ant_Maze_Args
from writer.init_writer import init_writer
from mpo.algorithm.__init__ import MPO
from mpo.train_script.MPO_Learner import MPO_Learner
from helpers.warm_up_compilation import warmup_mpo_compile
from mpo.evaluation_e_step.MPO_Learner_for_E_step import MPO_Learner_E_Step


ExperimentArgs = Union[
    Annotated[Ant_Maze_Args, tyro.conf.subcommand("ant_maze")],
    Annotated[Robust_Ant_Args, tyro.conf.subcommand("robust_ant")],
]

def make_envs(args, run_name):

    train_env = make_train_vec_env(
        args,
        args.env_id,
        args.seed,
        args.num_envs,
    )
    
    eval_env = make_eval_env(args, args.env_id, args.seed, capture_video = False, run_name = run_name, name_prefix = "eval")
    print("Env:", train_env)
    args.obs_dim   = int(np.prod(train_env.single_observation_space.shape))
    args.action_dim  = int(np.prod(train_env.single_action_space.shape))
    
    test_env = make_base_env(args.env_id, args, render_mode=None)
    args.action_space_low = test_env.action_space.low.copy()
    args.action_space_high = test_env.action_space.high.copy()
    test_env.close()
    print("Action dimensions:", args.action_dim)
    print("Observations dimensions:", args.obs_dim)
    print("action space:", args.action_space_low, args.action_space_high)
    
    return train_env, eval_env

def init_runname(args):
     # Run-Name à la CleanRL
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    return run_name

def make_networks(args, device):
    
    actor = Actor(args).to(device)
    critic = Critic(args).to(device)
    # Target networks (used for stable targets)
    target_actor = copy.deepcopy(actor).to(device)
    target_critic = copy.deepcopy(critic).to(device)
    return actor, critic, target_actor, target_critic

def make_replaybuffer(args):
    # Create ReplayBuffer
    if args.buffer_on_cuda and (args.device == "cuda" and torch.cuda.is_available()):
        device_buffer = "cuda"
    else: 
        device_buffer = "cpu"
    
    if args.episodic_replaybuffer:
        replaybuffer = EpisodicReplayBuffer(args.max_buffer_capacity, args.obs_dim, args.action_dim, device_buffer)
    else:
        replaybuffer = ReplayBuffer(args.max_buffer_capacity, args.obs_dim, args.action_dim, device_buffer)
    
    return replaybuffer
     
def make_optimizer(args, actor, critic):
    actor_optimizer = torch.optim.Adam(actor.parameters(), args.actor_lr)
    critic_optimizer = torch.optim.Adam(critic.parameters(), args.critic_lr)
    return actor_optimizer, critic_optimizer

def train():
    args = tyro.cli(ExperimentArgs)
    try: 
        num_threads = int(_DEFAULT_THREADS)
    except ValueError:
        num_threads = None
    limit_threads(num_threads)
    
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
    train_env.reset()  

    # Create Networks
    actor, critic, target_actor, target_critic = make_networks(args, device)

    # Create Optimizer
    actor_optimizer, critic_optimizer = make_optimizer(args, actor, critic)

    
    # Create Replaybuffer
    replaybuffer = make_replaybuffer(args)
   
    # Create MPO Policy Optimizer
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

    writer = init_writer(args)

    if getattr(args, "use_e_step_eval", False):
        MPO_Learner_E_Step(args, train_env, eval_env, device, replaybuffer, mpo, writer)
    else:
        MPO_Learner(args, train_env, eval_env, device, replaybuffer, mpo, writer)


if __name__ == "__main__":
    train()

  


