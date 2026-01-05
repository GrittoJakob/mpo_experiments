import gymnasium as gym
import torch
import numpy as np



def evaluate(args, actor, eval_env, writer, device, global_step):
    """
    Run evaluation episodes using the current policy (self.actor)
    and return the average total reward per episode.
    """
    # No gradients needed during evaluation
    with torch.no_grad():
        total_rewards = []
        episode_len = []
        action_list = []
        for ep_idx in range(args.evaluate_episode_num):
            total_reward = 0.0
            ep_steps= 0.0
            # Reset environment at the beginning of each episode
            state, info = eval_env.reset()
            for s in range(args.evaluate_episode_maxstep):

                # Convert state to tensor on the correct device
                state_tensor = torch.as_tensor(
                    state, dtype=torch.float32, device=device
                    )
                
                # Get action from actor
                action = actor.action(state_tensor, deterministic = True)
                action_list.append(action)

                # Step environment
                next_state, reward, terminated, truncated, info = eval_env.step(action)
                done = terminated or truncated
                ep_steps +=1

                # Accumulate reward
                total_reward += reward
                if done:
                    break
                state =  next_state
            total_rewards.append(total_reward)
            episode_len.append(ep_steps)

             # 3. Logging (Now robust and wrapper-independent)

    mean_return = float(np.mean(total_rewards))
    mean_episode_len = float(np.mean(episode_len))
    
    print(f"Eval Return: {mean_return:.2f}")
    writer.add_scalar("eval/episodic_return", mean_return, global_step)
    writer.add_scalar("eval/episodic_length", mean_episode_len, global_step)
    
    # Log Action Magnitude
    actions = np.concatenate(action_list, axis=0)   # shape [T, act_dim]
    mean_abs = np.mean(np.abs(actions))
    max_abs  = np.max(np.abs(actions))
    mean_raw = np.mean(actions)
    writer.add_scalar("eval/action_mean_abs", mean_abs, global_step)
    writer.add_scalar("eval/action_max_abs", max_abs, global_step)
    writer.add_scalar("eval/action_mean", mean_raw, global_step)



    # Average return over all evaluation episodes
    return mean_return, mean_episode_len
