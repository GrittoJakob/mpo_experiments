import random
import time
import os
import math
from tqdm import tqdm
from dataclasses import dataclass
import numpy as np
import gymnasium as gym
import torch
import torch.optim as optim
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter
from nets.actor import Actor
from nets.critic import Critic
from mpo.__init__ import MPO
from buffer.replaybuffer import ReplayBuffer
from rollout.rollout import collect_rollout
from rollout.video_rollout import log_one_episode_video
from train_mpo.evaluation import evaluate
from helpers.sample_minibatch import sample_minibatch, assert_batch_shapes
from helpers.store_trajectory import store_trajectory
from helpers.logging import logging_wandb



def train_loop(
        args, 
        train_env, 
        eval_env,
        device: torch.device,
        replaybuffer: ReplayBuffer,
        mpo: MPO,
        gpu_buffer: bool = False
        ):
    """
    Main training loop for MPO.
    - Collects experience from the environment.
    - Updates critic (policy evaluation).
    - Runs MPO E-step and M-step (policy improvement).
    - Periodically evaluates and logs statistics.
    """

    if args.wandb_track:
        import wandb
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=args.run_name,
            save_code=True,
        )
        wandb.define_metric("grad_updates")
        wandb.define_metric("video", step_metric="grad_upates")


    writer = SummaryWriter(f"runs/{args.run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )


    num_steps = 0
    it = 1
    grad_updates = 0
    @dataclass
    class Runtime: 
        rollout: float = 0.0
        E_step: float = 0.0
        M_step: float = 0.0
        Critic_update: float = 0.0
        Eval: float = 0.0
        Sample_from_buffer: float = 0.0
        Sample_actions: float = 0.0
        target_critic_forward: float = 0.0
        
    
  
   
    best_video_reward = 0.0

    print(f"[DEBUG] start with number of steps = {len(replaybuffer)}, maximal number of environment steps: {args.max_training_steps}")       

    
    # Warm-up: fill replay buffer with some initial experience
    while len(replaybuffer) < args.warm_up_steps:
        new_steps = collect_rollout(train_env, args, mpo.actor, replaybuffer, device, gpu_buffer)
        num_steps += new_steps

    
    # Main training iterations
    pbar = tqdm(total=args.max_training_steps, desc="Env steps")
 

    while num_steps < args.max_training_steps:

        # Collect fresh experience for this iteration
        t_env_start = time.perf_counter()
        new_steps = collect_rollout(train_env, args, mpo.actor, replaybuffer, device, gpu_buffer)

        #Update current steps for while loop
        num_steps += new_steps
        
        if args.capture_video and it % args.log_videos_period == 0:
            prefix = f"rollout_gu{grad_updates}"
            trajectory, video_reward =  log_one_episode_video(args, mpo.actor, device, prefix, num_steps)
            if video_reward > best_video_reward and args.track_trajectory:
                best_video_reward = video_reward
                store_trajectory(
                    trajectory,
                    args=args,
                    global_step=num_steps,
                    reward=video_reward,
                    name=f"best_video_traj_{prefix}",
                )



        # For terminal logging
        pbar.update(new_steps)

        # Logging runtime env end
        t_env_end = time.perf_counter()
        Runtime.rollout += t_env_end - t_env_start

        num_updates_per_iter = math.ceil(
        args.UTD_ratio * new_steps
        )


        # Perform several updates per iteration (UTD ratio)
        for i_update in range(num_updates_per_iter):
            # if hasattr(torch, "compiler") and hasattr(torch.compiler, "cudagraph_mark_step_begin"):
            #     torch.compiler.cudagraph_mark_step_begin()
            grad_updates += 1
            
            buffer_size= len(replaybuffer)

            # Sample mini batch from buffer
            sample_mbatch_start = time.time()
            state_batch, action_batch, next_state_batch, reward_batch, terminated_batch, truncated_batch = sample_minibatch(
                replaybuffer=replaybuffer,
                batch_size=args.batch_size,
                device=device,
                gpu_buffer=gpu_buffer,
            )
            
            #Check for correct shapes
            if (grad_updates-1) % 10000 == 0:
                assert_batch_shapes(state_batch, action_batch, next_state_batch, reward_batch, terminated_batch, truncated_batch,
                        args.batch_size, mpo.state_dim, mpo.action_dim)
            sample_mbatch_end = time.time()
            Runtime.Sample_from_buffer += sample_mbatch_end - sample_mbatch_start

            # Compute target_actor forward pass to get sampled_actions an b_mu, b_st
            # all_sampled_actions: sampled actions for timestep t and t+1 concatenated
            # sampled actions: sampled actions only for timestep t
            sample_action_start= time.time()
            all_sampled_actions, sampled_actions, mu_off, std_off = mpo.sample_actions_from_target_actor(
                state_batch= state_batch,
                next_state_batch= next_state_batch,
                sample_num= args.sample_action_num
            )
            sample_action_end = time.time()
            Runtime.Sample_actions += sample_action_end - sample_action_start

            # Compute forward pass of target critic for state_batch and next_state_batch
            t_critic_forward_start = time.time()
            target_q, next_target_q = mpo.target_critic_forward_pass(
                state_batch=state_batch,
                next_state_batch= next_state_batch,
                all_sampled_actions = all_sampled_actions
            )
            t_critic_forward_end = time.time()
            Runtime.target_critic_forward += t_critic_forward_end - t_critic_forward_start

            # Policy evaluation (critic update)
            t_policy_eval_start = time.perf_counter()
            collect_stats = args.wandb_track and (i_update % args.log_period == 0)
            critic_update_stats = mpo.critic_update_td( 
                next_target_q = next_target_q,
                state_batch =state_batch, 
                action_batch =action_batch, 
                reward_batch= reward_batch, 
                terminated_batch = terminated_batch,
                truncated_batch= truncated_batch,
                collect_stats= collect_stats
                )

            t_policy_eval_end = time.perf_counter()    
            Runtime.Critic_update += t_policy_eval_end - t_policy_eval_start


            if i_update % args.delay_policy_update == 0:
                
                # E-step (build non-parametric target distribution)
                t_E_step_start = time.perf_counter()
                norm_target_q, stats_e_step = mpo.expectation_step(
                    target_q= target_q,
                    sampled_actions=sampled_actions, 
                    collect_stats = collect_stats
                    )
                
                t_E_step_end = time.perf_counter()         
                Runtime.E_step += t_E_step_end - t_E_step_start

                # M-step (actor / policy update)
                t_M_step_start = time.perf_counter()
                stats_m_step = mpo.maximization_step(
                    state_batch=state_batch,
                    norm_target_q = norm_target_q, 
                    sampled_actions = sampled_actions, 
                    mu_off = mu_off, 
                    std_off = std_off,
                    collect_stats= collect_stats)
                t_M_step_end = time.perf_counter()
                Runtime.M_step += t_M_step_end - t_M_step_start


            # logging 
            if args.wandb_track and i_update % args.log_period == 0:
                logging_wandb(
                    writer = writer,
                    args = args,
                    replaybuffer = replaybuffer,
                    Runtime = Runtime, 
                    stats_m_step = stats_m_step,
                    stats_e_step = stats_e_step,
                    critic_update_stats = critic_update_stats,
                    grad_updates = grad_updates, 
                    num_steps = num_steps
                )

             
            # Target network update & logging
            # Periodically sync target networks with current actor/critic
            if grad_updates % args.target_update_period == 0:
                mpo.update_target_actor_critic()


        # Evaluation in the outer loop
        if it % args.evaluate_period == 0:
            
            t_eval_start = time.perf_counter()

            # Evaluate current policy without gradient tracking
            mpo.actor.eval()
            evaluate(args, mpo.actor, eval_env, writer, device, num_steps)
            mpo.actor.train()
            t_eval_end = time.perf_counter()
            Runtime.Eval += t_eval_end -t_eval_start 

        # if it % args.save_every == 0:
        #     mpo.save(it)

        it += 1  
        # Increase global update counter after each gradient update    
    
    mpo.actor.eval()
    mpo.critic.eval()
    with torch.no_grad():
        save_actor_critic(mpo, args, num_steps=num_steps, grad_updates=grad_updates, out_dir="checkpoints")
    mpo.actor.train()
    mpo.critic.train()
    pbar.close()

def save_actor_critic(mpo, args, num_steps: int, grad_updates: int, out_dir: str = "checkpoints"):
    """
    Speichert Actor + Critic gemeinsam (atomar) in einer .pt-Datei.
    Erwartet mpo.actor und mpo.critic.
    """
    os.makedirs(out_dir, exist_ok=True)

    payload = {
        "actor_state_dict": mpo.actor.state_dict(),
        "critic_state_dict": mpo.critic.state_dict(),
        "num_steps": int(num_steps),
        "grad_updates": int(grad_updates),
        "run_name": getattr(args, "run_name", None),
        "timestamp": time.strftime("%Y%m%d-%H%M%S"),
        "args": vars(args) if hasattr(args, "__dict__") else None,
    }

    filename = f"{payload['run_name'] or 'run'}_ac_steps{num_steps}_gu{grad_updates}.pt"
    final_path = os.path.join(out_dir, filename)
    tmp_path = final_path + ".tmp"

    torch.save(payload, tmp_path)
    os.replace(tmp_path, final_path)
    print(f"[SAVE] Actor+Critic saved to: {final_path}")



