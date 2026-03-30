import gymnasium as gym
import torch
import numpy as np


def collect_rollout(env, state, args, actor, replaybuffer, device):
    """
    Collect rollout steps with the current policy and store them in the replay buffer.

    Stores per transition:
        (obs, action, next_obs, reward, terminated, truncated)

    Returns:
        next_state: latest env state after rollout
        total_steps_collected: number of collected environment steps
    """

    num_envs = env.num_envs
    T = args.sample_steps_per_iter
    total_steps_collected = 0

    obs_buf = np.empty((T, num_envs, args.obs_dim), dtype=np.float32)
    actions_buf = np.empty((T, num_envs, args.action_dim), dtype=np.float32)
    next_obs_buf = np.empty((T, num_envs, args.obs_dim), dtype=np.float32)
    rewards_buf = np.empty((T, num_envs), dtype=np.float32)
    terminated_buf = np.empty((T, num_envs), dtype=np.float32)
    truncated_buf = np.empty((T, num_envs), dtype=np.float32)
    actor.eval()
    
    with torch.no_grad():
        
        for step in range(args.sample_steps_per_iter):
        
            # Convert state from numpy to tensor for forward pass
            state_tensor = torch.as_tensor(state, dtype=torch.float32, device=device)

            actions = actor.action(state_tensor)  # action is numpy 

            # Step
            next_states, rewards, terminated, truncated, infos = env.step(actions)

            # Handle dones and final observation for async vector env:
            # If terminated or truncated: get next_obs from final_observations
            done_mask = np.logical_or(terminated, truncated)

            # Copy next_states for next step
            real_next_obs = next_states.copy()

            if "final_observation" in infos:
                for idx_env, done in enumerate(done_mask):
                    if done:
                        real_next_obs[idx_env] = infos["final_observation"][idx_env] 


            obs_buf[step] = state
            actions_buf[step] = actions
            next_obs_buf[step] = real_next_obs
            rewards_buf[step] = rewards
            terminated_buf[step] = terminated.astype(np.float32)
            truncated_buf[step] = truncated.astype(np.float32)
            
            state = next_states
            total_steps_collected += num_envs

    actor.train()

    # Flatten for Buffer
    obs_flat = obs_buf.reshape(T * num_envs, args.obs_dim)
    actions_flat = actions_buf.reshape(T * num_envs, args.action_dim)
    next_obs_flat = next_obs_buf.reshape(T * num_envs, args.obs_dim)
    rewards_flat = rewards_buf.reshape(T * num_envs, 1)
    truncated_flat = truncated_buf.reshape(T * num_envs, 1)
    terminated_flat = terminated_buf.reshape(T * num_envs, 1)

    replaybuffer.add_batch(
        obs = obs_flat,
        actions = actions_flat,
        next_obs = next_obs_flat,
        rewards = rewards_flat,
        terminated = terminated_flat,
        truncated = truncated_flat
    )

    return state, total_steps_collected
