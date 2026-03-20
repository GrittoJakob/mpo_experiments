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
from mpo.__init__ import MPO
from buffer.replaybuffer import ReplayBuffer
from rollout.rollout import collect_rollout
from rollout.video_rollout import log_one_episode_video
from mpo_baseline.runners.evaluation import evaluate
from mpo_baseline.buffer.sample_minibatch import sample_minibatch, assert_batch_shapes
from helpers.store_trajectory import store_trajectory
from mpo_baseline.writer.logging import logging_wandb
from helpers.save_model import save_actor_critic



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
    
    # Warm-up: fill replay buffer with some initial experience
    while len(replaybuffer) < args.warm_up_steps:
        new_steps = collect_rollout(train_env, args, mpo.actor, replaybuffer, device, gpu_buffer)
        num_steps += new_steps

    
    # Main training iterations
    pbar = tqdm(total=args.max_training_steps, desc="Env steps")
    while num_steps < args.max_training_steps:

        # Collect fresh experience for this iteration
        new_steps = collect_rollout(train_env, args, mpo.actor, replaybuffer, device, gpu_buffer)

        #Update current steps for while loop
        num_steps += new_steps

        # Capture Video        
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

        # Number of updates dependant from UTD ratio and collected steps during rollout
        num_updates_per_iter = math.ceil(
            args.UTD_ratio * new_steps
            )

        # Perform several updates per iteration (UTD ratio)
        for i_update in range(num_updates_per_iter):
            grad_updates += 1

            # Sample mini batch from buffer
            state_batch, action_batch, next_state_batch, reward_batch, terminated_batch, truncated_batch = sample_minibatch(
                replaybuffer=replaybuffer,
                batch_size=args.batch_size,
                device=device,
                gpu_buffer=gpu_buffer,
            )

            # Compute target_actor forward pass to get sampled_actions an b_mu, b_st
            # all_sampled_actions: sampled actions for timestep t and t+1 concatenated
            # sampled actions: sampled actions only for timestep t
            all_sampled_actions, sampled_actions, mu_off, std_off = mpo.sample_actions_from_target_actor(
                state_batch= state_batch,
                next_state_batch= next_state_batch,
                sample_num= args.sample_action_num
            )

            # Compute forward pass of target critic for state_batch and next_state_batch
            target_q, next_target_q = mpo.shared_target_critic_forward_pass(
                state_batch=state_batch,
                next_state_batch= next_state_batch,
                all_sampled_actions = all_sampled_actions
            )

            # Policy evaluation (critic update)
            collect_stats = args.wandb_track and (i_update % args.log_period == 0) and (i_update & args.delay_policy_update == 0)
            critic_update_stats = mpo.critic_update_td( 
                next_target_q = next_target_q,
                state_batch =state_batch, 
                action_batch =action_batch, 
                reward_batch= reward_batch, 
                terminated_batch = terminated_batch,
                truncated_batch= truncated_batch,
                collect_stats= collect_stats
                )

            # Delay policy updates for better stability by training the critic more often
            if (i_update % args.delay_policy_update == 0):
                
                # E-step (build non-parametric target distribution)
                norm_target_q, stats_e_step = mpo.expectation_step(
                    target_q= target_q,
                    sampled_actions=sampled_actions, 
                    collect_stats = collect_stats
                    )

                # M-step (actor / policy update)
                stats_m_step = mpo.maximization_step(
                    state_batch=state_batch,
                    norm_target_q = norm_target_q, 
                    sampled_actions = sampled_actions, 
                    mu_off = mu_off, 
                    std_off = std_off,
                    collect_stats= collect_stats)


            # logging 
            if args.wandb_track and (i_update % args.log_period == 0) and collect_stats:
                logging_wandb(
                    writer = writer,
                    args = args,
                    replaybuffer = replaybuffer, 
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


        # Evaluation
        if it % args.evaluate_period == 0:

            # Evaluate current policy without gradient tracking
            mpo.actor.eval()
            evaluate(args, mpo.actor, eval_env, writer, device, num_steps)
            mpo.actor.train()

        # Update iteration counter
        it += 1    
    
    # Save the actor and critic after training
    mpo.actor.eval()
    mpo.critic.eval()
    with torch.no_grad():
        save_actor_critic(mpo, args, num_steps=num_steps, grad_updates=grad_updates, out_dir="checkpoints")
    mpo.actor.train()
    mpo.critic.train()
    pbar.close()

