import gymnasium as gym
import torch
import numpy as np
import options


def evaluate(args, actor, eval_env, writer, device, global_step):
    """
    Run evaluation episodes using the current policy (self.actor)
    and return the average total reward per episode.
    """


    with torch.no_grad():
        total_rewards = []
        episode_len = []
        action_list = []
        if args.task_mode == "inverted" or "inverted_multi_task":
            eval_tasks = [{"task_mode": 1.0}, {"task_mode": -1.0}]
        else:
            eval_tasks = [None]
        # if args.rand_mode == "RFI":
        #     options = {}
        #     curr_rand_noise = 0.0
        #     eval_tasks= [{"rao_limit": 0.0}]

        for ep_idx in range(args.evaluate_episode_num):
            for task_options in eval_tasks:
                
                options = task_options 
                total_reward = 0.0
                ep_steps= 0.0
                # Reset environment at the beginning of each episode
                
                state, info = eval_env.reset(options = options)
                for s in range(args.evaluate_episode_maxstep):

                    # Convert state to tensor on the correct device
                    state_tensor = torch.as_tensor(
                        state, dtype=torch.float32, device=device
                        )
                    
                    # Get action from actor
                    action = actor.action(state_tensor, deterministic = True)
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

                if args.task_mode == "target_goal" and task_options is not None:
                    gx, gy = task_options["target_goal"]
                    # Tag
                    tag = f"eval/return_goal_x{gx:.1f}_y{gy:.1f}".replace("-", "m").replace(".", "p")
                    writer.add_scalar(tag, total_reward, global_step)
                elif args.task_mode == "inverted_multi_task":
                    sign = task_options["task_mode"]
                    # Tag:
                    tag = f"eval/return_direction_{sign:.1f}"
                

                # 3. Logging (Now robust and wrapper-independent)

    mean_return = float(np.mean(total_rewards))
    mean_episode_len = float(np.mean(episode_len))
    
    print(f"Eval Return: {mean_return:.2f}")
    writer.add_scalar("eval/episodic_return", mean_return, global_step)
    writer.add_scalar("eval/episodic_length", mean_episode_len, global_step)
    
    # Log Action Magnitude
    actions = np.stack(action_list, axis=0)  # [T, act_dim]
    mean_abs = np.mean(np.abs(action_list))
    max_abs  = np.max(np.abs(action_list))
    mean_raw = np.mean(action_list)
    writer.add_scalar("eval/action_mean_abs", mean_abs, global_step)
    writer.add_scalar("eval/action_max_abs", max_abs, global_step)
    writer.add_scalar("eval/action_mean", mean_raw, global_step)



    # Average return over all evaluation episodes
    return mean_return, mean_episode_len
