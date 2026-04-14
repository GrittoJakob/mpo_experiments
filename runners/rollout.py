import torch
import numpy as np


def collect_rollout(env, state, unfinished_episodes, args, actor, replaybuffer, device):
    """
    Collect rollout steps with the current policy and store them in the replay buffer.

    Stores per transition:
        (obs, action, next_obs, reward, terminated, truncated)

    Returns:
        dict of unfinished episodes: latest env state after rollout
        total_steps_collected: number of collected environment steps
    """

    num_envs = env.num_envs
    T = args.sample_steps_per_iter
    total_steps_collected = 0
    
    # For first init of unfinished episodes
    if unfinished_episodes is None:
        unfinished_episodes = [empty_episode() for _ in range(num_envs)]

    # Check shapes
    assert len(unfinished_episodes) == num_envs, (
        f"unfinished_episodes must have length {num_envs}, "
        f"got {len(unfinished_episodes)}"
    )

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
                for env_idx, done in enumerate(done_mask):
                    if done:
                        real_next_obs[env_idx] = infos["final_observation"][env_idx] 


            for env_idx in range(num_envs):
                        
                        # Iterate over environments
                        ep = unfinished_episodes[env_idx]

                        ep["obs"].append(state[env_idx].copy())
                        ep["actions"].append(actions[env_idx].copy())
                        ep["next_obs"].append(real_next_obs[env_idx].copy())
                        ep["rewards"].append(rewards[env_idx])
                        ep["terminated"].append(terminated[env_idx])
                        ep["truncated"].append(truncated[env_idx])

                        if done_mask[env_idx]:

                            replaybuffer.add_batch(
                                obs=np.asarray(ep["obs"], dtype=np.float32),
                                actions=np.asarray(ep["actions"], dtype=np.float32),
                                next_obs=np.asarray(ep["next_obs"], dtype=np.float32),
                                rewards=np.asarray(ep["rewards"], dtype=np.float32),
                                terminated=np.asarray(ep["terminated"], dtype=np.float32),
                                truncated=np.asarray(ep["truncated"], dtype=np.float32),
                            )
                            # Clearn finished epsiode again
                            unfinished_episodes[env_idx] = empty_episode()

            state = next_states
            total_steps_collected += num_envs
            step += num_envs

    actor.train()

    return state, unfinished_episodes, total_steps_collected


def empty_episode():
    return {
        "obs": [],
        "actions": [],
        "next_obs": [],
        "rewards": [],
        "terminated": [],
        "truncated": [],
    }

