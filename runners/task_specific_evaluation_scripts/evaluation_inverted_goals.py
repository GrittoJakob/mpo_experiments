import torch
import numpy as np

def evaluate_inverted_goal(args, actor, eval_env, writer, device, global_step):
    """
    Run evaluation episodes using the current policy (self.actor)
    and return the average total reward per episode.
    """

    with torch.no_grad():
        total_rewards = []
        episode_len = []
        action_list = []
        rewards_by_task = {}    
        len_by_task = {}
        actions_by_task = {}
        task_reward_by_task = {}
        eval_tasks = [{"task_mode": 1.0}, {"task_mode": -1.0}]

        for task_options in eval_tasks:
            options = task_options 
            task_mode = options["task_mode"]

            rewards_by_task.setdefault(task_mode, [])
            task_reward_by_task.setdefault(task_mode, [])
            len_by_task.setdefault(task_mode, [])

            for ep_idx in range(args.evaluate_episode_num):
                
                total_reward = 0.0
                ep_steps= 0
                # Reset environment at the beginning of each episode
                
                state, info = eval_env.reset(options = options)
                while True:

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
                    task_reward = info['velocity_reward']

                    # Accumulate reward
                    total_task_rewards = task_reward
                    total_reward += reward
                    if done:
                        break
                    state =  next_state

                # Append episode to list
                total_rewards.append(total_reward)
                episode_len.append(ep_steps)
                rewards_by_task[task_mode].append(total_reward)
                len_by_task[task_mode].append(ep_steps)
                task_reward_by_task[task_mode].append(total_task_rewards)


    mean_return = float(np.mean(total_rewards))
    mean_episode_len = float(np.mean(episode_len))
    
    print(f"Eval Return: {mean_return:.2f}")
    writer.add_scalar("eval/episodic_return", mean_return, global_step)
    writer.add_scalar("eval/episodic_length", mean_episode_len, global_step)

    # Task specific:
    mean_return_task1 = float(np.mean(rewards_by_task[1.0]))
    mean_return_task2 = float(np.mean(rewards_by_task[-1.0]))
    mean_task_return_task1 = float(np.mean(task_reward_by_task[1.0]))
    mean_task_return_task2 = float(np.mean(task_reward_by_task[-1.0]))
    writer.add_scalar("eval/episodic_return_task_forward", mean_return_task1, global_step)
    writer.add_scalar("eval/episodic_return_task_backward", mean_return_task2, global_step)
    writer.add_scalar("eval/episodic_length_task_forward", float(np.mean(len_by_task[1.0])), global_step)
    writer.add_scalar("eval/episodic_length_task_backward", float(np.mean(len_by_task[-1.0])), global_step)
    writer.add_scalar("eval/episodic_task_return_task_forward", mean_task_return_task1)
    writer.add_scalar("eval/episodic_task_return_task_backward", mean_task_return_task2)
