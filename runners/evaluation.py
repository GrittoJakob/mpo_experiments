import gymnasium as gym
import torch
import numpy as np
import options
from .task_specific_evaluation_scripty.evaluation_inverted_goals import evaluate_inverted_goal
from .task_specific_evaluation_scripty.evaluation_ERFI_noise import evaluate_erfi
from .task_specific_evaluation_scripty.evaluation_target_goals import evaluate_target_goal

def evaluate(args, actor, eval_env, writer, device, global_step):
    """
    Run evaluation episodes using the current policy (self.actor)
    and return the average total reward per episode.
    """

    if args.task_mode in ["inverted_without_task_hint"]:
        return evaluate_inverted_goal(args, actor, eval_env, writer, device, global_step)
    elif args.task_mode in ("target_goal"):
        return evaluate_target_goal(args, actor, eval_env, writer, device, global_step)
    elif args.rand_mode == "ERFI":
        return evaluate_erfi(args, actor, eval_env, writer, device, global_step)

    
    with torch.no_grad():
        total_rewards = []
        episode_len = []
        action_list = []

        # Iterate over defined number of eval episodes
        for ep_idx in range(args.evaluate_episode_num):
                
                # Set running variables to zero
                total_reward = 0.0
                ep_steps= 0.0

                # Reset environment at the beginning of each episode
                state, info = eval_env.reset()

                # Loop over one episode
                for s in range(args.evaluate_episode_maxstep):

                    # Convert state to tensor on the correct device
                    state_tensor = torch.as_tensor(
                        state, dtype=torch.float32, device=device
                        )
                    
                    # Get action from actor
                    action = actor.action(state_tensor, clip_to_env = True, deterministic = True)
                    if args.use_action_clipping:
                        action = np.clip(action, args.action_space_low, args.action_space_high)
                    
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
      

    mean_return = float(np.mean(total_rewards))
    mean_episode_len = float(np.mean(episode_len))
    
    print(f"Eval Return: {mean_return:.2f}")
    writer.add_scalar("eval/episodic_return", mean_return, global_step)
    writer.add_scalar("eval/episodic_length", mean_episode_len, global_step)
    

