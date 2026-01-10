import gymnasium as gym
import torch
import numpy as np


def collect_rollout(env, args, actor, replaybuffer, device, buffer_gpu):
        """
        Collect a number of episodes by interacting with the environment
        using the current policy and store them in the replay buffer.
        Each episode is stored as a list of (state, action, next_state, reward) tuples.
        Returns:
        total_steps_collected (int): number of env steps actually executed
        """
        episodes = []
        total_steps_collected = 0
        actor.eval()
        
        with torch.no_grad():
            # Loop over episodes to collect
            while total_steps_collected < args.sample_steps_per_iter:
                
                if buffer_gpu: 
                    states_list, acts_list, next_states_list, rew_list = [], [], [], []
                else:
                    buff = []
                state, info = env.reset()

                # Roll out one episode up to a maximum number of steps
                for _ in range(args.sample_episode_maxstep):
                    state_tensor = torch.as_tensor(state, dtype=torch.float32, device=device)
                    
                    # Get action from actor
                    # NOTE: Actor.action already returns a NumPy array
                    action = actor.action(state_tensor)
                    if args.use_action_clipping:
                        action = np.clip(action, args.action_space_low, args.action_space_high)

                    # Step the environment
                    next_state, reward, terminated, truncated, info = env.step(action)
                    total_steps_collected += 1
                    
                    if buffer_gpu: 
                        states_list.append(state)
                        acts_list.append(action)           # numpy
                        next_states_list.append(next_state)
                        rew_list.append(reward)
                    else:
                        buff.append((state, action, next_state, reward))

                    if terminated or truncated:
                        break
                    state = next_state

                if buffer_gpu:    
                    states_np      = np.asarray(states_list,      dtype=np.float32)
                    actions_np     = np.asarray(acts_list,        dtype=np.float32)
                    next_states_np = np.asarray(next_states_list, dtype=np.float32)
                    rewards_np     = np.asarray(rew_list,        dtype=np.float32)

                    # Store completed episode
                    replaybuffer.store_episode_stacked(states_np, actions_np, next_states_np, rewards_np)
                else:
                    episodes.append(buff)

        actor.train()

        # Push all collected episodes into the replay buffer and returns the number of collected steps
        if buffer_gpu is False:
            replaybuffer.store_episodes(episodes)
        return total_steps_collected
