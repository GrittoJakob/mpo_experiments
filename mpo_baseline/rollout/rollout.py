import gymnasium as gym
import torch
import numpy as np


def collect_rollout(env, args, actor, replaybuffer, device, buffer_gpu):
    """
    Collect rollouts with the current policy and store them in the replay buffer.

    Stores per transition:
      (state, action, next_state, reward, terminated, truncated)

    Returns:
      total_steps_collected (int)
    """
    episodes_cpu = []  # only used when buffer_gpu == False
    total_steps_collected = 0
    actor.eval()

    with torch.no_grad():
        while total_steps_collected < args.sample_steps_per_iter:
            # Per-episode storage (always lists -> no branching in the step loop)
            states_list, acts_list, next_states_list = [], [], []
            rews_list, terminated_list, truncated_list = [], [], []
            task_invert_list, vel_rew_list = [], []


            state, info = env.reset()
            task_invert = info['task_direction']

            for _ in range(args.sample_episode_maxstep):
                state_tensor = torch.as_tensor(state, dtype=torch.float32, device=device)

                action = actor.action(state_tensor)  # numpy
                if args.use_action_clipping:
                    action = np.clip(action, args.action_space_low, args.action_space_high)

                next_state, reward, terminated, truncated, info = env.step(action)
                total_steps_collected += 1
                
                # velocity rewards
                vel_rew = info['velocity_reward']
                if args.task_mode == "inverted":
                  
                    task_invert = info['task_direction']
                else:
                    task_invert = 1.0
            

                # Store transition parts
                states_list.append(state)
                acts_list.append(action)
                next_states_list.append(next_state)
                rews_list.append(reward)
                terminated_list.append(terminated)
                truncated_list.append(truncated)
                task_invert_list.append(task_invert)
                vel_rew_list.append(vel_rew)

              

                done = terminated or truncated
                if done:
                    break

                state = next_state

            # Episode finished -> store once, depending on buffer type
            if buffer_gpu:
                states_np           = np.asarray(states_list,      dtype=np.float32)
                actions_np          = np.asarray(acts_list,        dtype=np.float32)
                next_states_np      = np.asarray(next_states_list, dtype=np.float32)
                rewards_np          = np.asarray(rews_list,        dtype=np.float32)
                terminated_np       = np.asarray(terminated_list,  dtype=np.float32)
                truncated_np        = np.asarray(truncated_list,   dtype=np.float32)
                task_invert_np      = np.asarray(task_invert_list, dtpye=np.float32)
                vel_rew_np          = np.asarray(vel_rew_list, dtype=np.float32) 
                    
                replaybuffer.store_episode_stacked(
                states_np, actions_np, next_states_np, rewards_np, terminated_np, truncated_np, task_invert_np, vel_rew_np
                )
                
            else:
                # Build episode as list of tuples (CPU buffer)
            
                episode = list(zip(
                    states_list, acts_list, next_states_list, rews_list, terminated_list, truncated_list, task_invert_list, vel_rew_list
                    ))
                
                episodes_cpu.append(episode)

    actor.train()

    if not buffer_gpu:
        replaybuffer.store_episodes(episodes_cpu)

    return total_steps_collected
