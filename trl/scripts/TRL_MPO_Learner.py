import math
from tqdm import tqdm
from dataclasses import dataclass
import torch
from torch.utils.tensorboard import SummaryWriter
from trl.algorithm.__init__ import TRL_with_MPO
from runners.rollout import collect_rollout
from runners.video_rollout import log_one_episode_video
from runners.evaluation import evaluate
from writer.logging import logging_wandb
from helpers.save_model import save_actor_critic



def TRL_with_MPO_Learner(
        args, 
        train_env, 
        eval_env,
        device: torch.device,
        replaybuffer,
        trl_mpo: TRL_with_MPO,
        writer: SummaryWriter,
        ):
    """
    Main training loop for MPO.
    - Collects experience from the environment.
    - Updates critic (policy evaluation).
    - Runs MPO E-step and M-step (policy improvement).
    - Periodically evaluates and logs statistics.
    """
    # Initialisation for rollout:
    state, _ = train_env.reset()
    unfinished_episodes = None
    num_steps = 0

    # Iterators
    it = 0
    grad_updates = 0 
    
    # Warm-up: fill replay buffer with some initial experience
    while len(replaybuffer) < args.warm_up_steps:
        state, unfinished_episodes, new_steps = collect_rollout(train_env, state, unfinished_episodes, args, trl_mpo.actor, replaybuffer, device)
        num_steps += new_steps

    
    # Main training iterations
    pbar = tqdm(total=args.max_training_steps, desc="Env steps")
    while num_steps < args.max_training_steps:

        # Collect fresh experience for this iteration
        state, unfinished_episodes, new_steps = collect_rollout(train_env, state, unfinished_episodes, args, trl_mpo.actor, replaybuffer, device)

        # Update current steps for while loop
        num_steps += new_steps

        # Capture Video        
        if args.capture_video and it % args.log_videos_period == 0:
            prefix = f"rollout_gu{grad_updates}"
            log_one_episode_video(args, trl_mpo.actor, device, prefix, grad_updates)

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
            batch = replaybuffer.sample_batch(args.batch_size)
            obs_batch = batch["start_obs"]
            actions_batch = batch["actions"]

            # Policy evaluation (critic update)
            collect_stats = (args.wandb_track 
                and (i_update % args.log_period == 0) 
                and (i_update % args.delay_policy_update == 0))
            
            ## TRL Critic Update
            critic_update_stats = trl_mpo.trl_critic_loss(batch, collect_stats)

            # Delay policy updates for better stability by training the critic more often
            if (i_update % args.delay_policy_update == 0):
                
                # sample actions
                sampled_actions, mu_off, std_off = trl_mpo.sample_actions_from_target_actor(
                state_batch= obs_batch,
                sample_num= args.sample_action_num
                )

                # Compute Target critic for E-Step
                target_q = trl_mpo.target_critic_forward_pass(obs_batch, sampled_actions)
                
                # E-step (build non-parametric target distribution)
                norm_target_q, stats_e_step = trl_mpo.expectation_step(
                    target_q= target_q,
                    sampled_actions=sampled_actions, 
                    collect_stats = collect_stats
                    )

                # M-step (actor / policy update)
                stats_m_step = trl_mpo.maximization_step(
                    state_batch= obs_batch,
                    norm_target_q = norm_target_q, 
                    sampled_actions = sampled_actions, 
                    mu_off = mu_off, 
                    std_off = std_off,
                    collect_stats= collect_stats)


            # logging to wandb
            if args.wandb_track and (i_update % args.log_period == 0) and collect_stats:
                logging_wandb(
                    writer = writer,
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
                trl_mpo.update_target_actor_critic()


        # Evaluation
        if it % args.evaluate_period == 0:

            # Evaluate current policy without gradient tracking
            trl_mpo.actor.eval()
            evaluate(args, trl_mpo.actor, eval_env, writer, device, grad_updates)
            trl_mpo.actor.train()

        # Update iteration counter
        it += 1    
    
    # Save the actor and critic after training
    trl_mpo.actor.eval()
    trl_mpo.critic.eval()
    with torch.no_grad():
        save_actor_critic(trl_mpo, args, num_steps=num_steps, grad_updates=grad_updates, out_dir="checkpoints")
    trl_mpo.actor.train()
    trl_mpo.critic.train()
    pbar.close()
