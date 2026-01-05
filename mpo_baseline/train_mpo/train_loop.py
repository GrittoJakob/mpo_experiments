import random
import time
import os
import math
from tqdm import tqdm
import numpy as np
import gymnasium as gym
import torch
import torch.optim as optim
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter
from nets.actor import Actor
from nets.critic import Critic
from mpo.mpo import MPO
from buffer.replaybuffer import ReplayBuffer
from rollout.rollout import collect_rollout
from rollout.video_rollout import log_one_episode_video
from train_mpo.evaluation import evaluate



def train_loop(
        args, 
        train_env, 
        eval_env,
        device: torch.device,
        replaybuffer: ReplayBuffer,
        mpo: MPO
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


    max_training_steps = args.max_training_steps
    num_steps = 0
    it = 0
    grad_updates = 0
    runtime_rollout = 0.0
    runtime_E_step = 0.0
    runtime_M_step = 0.0
    runtime_policy_eval = 0.0
    runtime_eval = 0.0

    print(f"[DEBUG] start with number of steps = {len(replaybuffer)}, maximal number of environment steps: {max_training_steps}")       

    # Buffer size for warm up
    buffer_size = replaybuffer._current_size()
    
    # Warm-up: fill replay buffer with some initial experience
    while buffer_size < args.warm_up_steps:
        new_steps = collect_rollout(train_env, args, mpo.actor, replaybuffer, device)
        num_steps += new_steps

        # Update buffer size after adding new episodes
        buffer_size= replaybuffer._current_size()
    
    # Main training iterations
    pbar = tqdm(total=max_training_steps, desc="Env steps")
 

    while num_steps < max_training_steps:

        # Collect fresh experience for this iteration
        t_env_start = time.perf_counter()
        new_steps = collect_rollout(train_env, args, mpo.actor, replaybuffer, device)

        #Update current steps for while loop
        num_steps += new_steps
        
        if args.capture_video and it % args.log_videos_period == 0:
            prefix = f"rollout_gu{grad_updates}"
            log_one_episode_video(args, mpo.actor, device, prefix, num_steps)

        # For terminal logging
        pbar.update(new_steps)

        # Logging runtime env end
        t_env_end = time.perf_counter()
        runtime_rollout += t_env_end - t_env_start

        num_updates_per_iter = math.ceil(
        args.UTD_ratio * new_steps
        )

        # Perform several updates per iteration (UTD ratio)
        for i_update in range(num_updates_per_iter):
            grad_updates += 1
            
            buffer_size= replaybuffer._current_size()
            # Sample a minibatch from buffer    
            indices = np.random.choice(
                buffer_size,
                size=args.batch_size,
                replace=False  # oder True, wenn du sehr viele Updates machen willst
            )        
            
            # Unpack transitions sampled from replay buffer    
            s_chunks, a_chunks, ns_chunks, r_chunks = replaybuffer.sample_batch(args.batch_size)

            state_batch      = torch.as_tensor(np.concatenate(s_chunks, axis=0), dtype=torch.float32, device=device)
            action_batch     = torch.as_tensor(np.concatenate(a_chunks, axis=0), dtype=torch.float32, device=device)
            next_state_batch = torch.as_tensor(np.concatenate(ns_chunks, axis=0), dtype=torch.float32, device=device)
            reward_batch     = torch.as_tensor(np.concatenate(r_chunks, axis=0), dtype=torch.float32, device=device)

            # Policy evaluation (critic update)
            t_policy_eval_start = time.perf_counter()

           
            critic_update_stats, sampled_actions, b_mu, b_std= mpo.critic_update_td( 
                state_batch =state_batch, 
                action_batch =action_batch, 
                next_state_batch =next_state_batch, 
                reward_batch= reward_batch, 
                sample_num =args.sample_action_num
                )

            t_policy_eval_end = time.perf_counter()    
            runtime_policy_eval += t_policy_eval_end - t_policy_eval_start

            if i_update % args.delay_policy_update == 0:
                
                # E-step (build non-parametric target distribution)
                t_E_step_start = time.perf_counter()
                sampled_actions, norm_target_q, b_mu, b_A, stats_e_step = mpo.expectation_step(
                    state_batch =state_batch, 
                    sampled_actions=sampled_actions, 
                    b_mu = b_mu, 
                    b_std = b_std
                    )
                
                t_E_step_end = time.perf_counter()         
                runtime_E_step += t_E_step_end - t_E_step_start

                # M-step (actor / policy update)
                t_M_step_start = time.perf_counter()
                stats_m = mpo.maximization_step(
                    state_batch=state_batch,
                    norm_target_q = norm_target_q, 
                    sampled_actions = sampled_actions, 
                    b_mu = b_mu, 
                    b_std = b_std)
                t_M_step_end = time.perf_counter()
                runtime_M_step += t_M_step_end - t_M_step_start


            # logging 
            if args.wandb_track and i_update % args.log_period == 0:
                
                # Compute mean reward/return in replay buffer
                mean_reward_buffer = replaybuffer.mean_reward()
                mean_return_buffer = replaybuffer.mean_return() 
                mean_episode_len = mean_return_buffer/(mean_reward_buffer+ 1e-8)                
                
                # Timing
                writer.add_scalar("time/rollout_time_sec", runtime_rollout, grad_updates)
                writer.add_scalar("time/E-Step", runtime_E_step, grad_updates)
                writer.add_scalar("time/M-Step", runtime_M_step, grad_updates)
                writer.add_scalar("time/critic_update", runtime_policy_eval, grad_updates)
                writer.add_scalar("time/evaluation", runtime_eval, grad_updates)
                
                writer.add_scalar("charts/learning_rate_actor", mpo.actor_optimizer.param_groups[0]["lr"], grad_updates)
                writer.add_scalar("charts/learning_rate_critic", mpo.critic_optimizer.param_groups[0]["lr"], grad_updates)

                #  Buffer
                writer.add_scalar("buffer/size", replaybuffer._current_size() ,grad_updates)
                writer.add_scalar("buffer/mean_return", mean_return_buffer , grad_updates)
                writer.add_scalar("buffer/mean_reward_per_step", mean_reward_buffer ,grad_updates)
                writer.add_scalar("buffer/mean_episode_len", mean_episode_len, grad_updates)
                writer.add_scalar("buffer/total_num_steps",num_steps, grad_updates)
                
                # M-Step Loggingd
                writer.add_scalar("m-step/loss_p", stats_m["loss_p"], grad_updates)
                writer.add_scalar("m-step/loss_l", stats_m["loss_l"], grad_updates)
                writer.add_scalar("m-step/C_mu_mean", stats_m["C_mu_mean"], grad_updates)
                writer.add_scalar("m-step/C_sigma_mean", stats_m["C_sigma_mean"], grad_updates)
                writer.add_scalar("m-step/eta_mu_mean", stats_m["eta_mu_mean"], grad_updates)
                writer.add_scalar("m-step/eta_mu_max", stats_m["eta_mu_max"], grad_updates)
                writer.add_scalar("m-step/eta_mu_min", stats_m["eta_mu_min"], grad_updates)
                writer.add_scalar("m-step/eta_sigma_mean", stats_m["eta_sigma_mean"], grad_updates)
                writer.add_scalar("m-step/eta_sigma_min", stats_m["eta_sigma_min"], grad_updates)
                writer.add_scalar("m-step/eta_sigma_max", stats_m["eta_sigma_max"], grad_updates)
                writer.add_scalar("m-step/mu_mean", stats_m["mu_mean"], grad_updates)
                writer.add_scalar("m-step/std_mean", stats_m["std_mean"], grad_updates)
                writer.add_scalar("m-step/std_min", stats_m["std_min"], grad_updates)
                writer.add_scalar("m-step/std_max", stats_m["std_max"], grad_updates)
                writer.add_scalar("m-step/mu_min", stats_m["mu_min"], grad_updates)
                writer.add_scalar("m-step/mu_max", stats_m["mu_max"], grad_updates)
                
                # E-Step Logging
                writer.add_scalar("e-step/eta_dual", stats_e_step["eta_dual"], grad_updates)
                writer.add_scalar("e-step/eta_penalty", stats_e_step["eta_penalty"], grad_updates)
                writer.add_scalar("e-step/penalty_mean", stats_e_step["penalty_mean"], grad_updates)
                writer.add_scalar("e-step/penalty_min", stats_e_step["penalty_min"], grad_updates)
                writer.add_scalar("e-step/penalty_max", stats_e_step["penalty_max"], grad_updates)

                # Retrace Logging
                writer.add_scalar("critic_update/q_loss", critic_update_stats["critic_loss"], grad_updates)
                writer.add_scalar("critic_update/q_current_mean", critic_update_stats["q_current_mean"], grad_updates)
                writer.add_scalar("critic_update/q_target_mean", critic_update_stats["q_target_mean"], grad_updates)


        # Evaluation in the outer loop
        if it % args.evaluate_period == 0:
            
            t_eval_start = time.perf_counter()

            # Evaluate current policy without gradient tracking
            mpo.actor.eval()
            evaluate(args, mpo.actor, eval_env, writer, device, num_steps)
            mpo.actor.train()
            t_eval_end = time.perf_counter()
            runtime_eval += t_eval_end -t_eval_start 

        # Target network update & logging
        # Periodically sync target networks with current actor/critic
        if grad_updates % args.target_update_period == 0:
            mpo.update_target_actor_critic()

        # if it % args.save_every == 0:
        #     mpo.save(it)

        it += 1  
        # Increase global update counter after each gradient update    
    

    pbar.close()
