import gymnasium as gym
import torch
import numpy as np
import options


def evaluate(args, actor, eval_env, writer, device, global_step):
    """
    Run evaluation episodes using the current policy (self.actor)
    and return the average total reward per episode.
    """

    if args.task_mode in ["inverted", "inverted_multi_task"]:
        return evaluate_inverted(args, actor, eval_env, writer, device, global_step)

    if args.task_mode == "target_goal":
        return evaluate_target_goal(args, actor, eval_env, writer, device, global_step)

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

def evaluate_inverted(args, actor, eval_env, writer, device, global_step):
    """
    Run evaluation episodes using the current policy (self.actor)
    and return the average total reward per episode.
    """


    with torch.no_grad():
        total_rewards = []
        episode_len = []
        action_list = []
        rewards_by_task = {}     # z.B. {1.0: [..episoden..], -1.0: [..]}
        len_by_task = {}
        actions_by_task = {}
        eval_tasks = [{"task_mode": 1.0}, {"task_mode": -1.0}]

        for task_options in eval_tasks:
            options = task_options 
            task_mode = options["task_mode"]

            rewards_by_task.setdefault(task_mode, [])
            len_by_task.setdefault(task_mode, [])
            actions_by_task.setdefault(task_mode, [])

            for ep_idx in range(args.evaluate_episode_num):
                
                total_reward = 0.0
                ep_steps= 0
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
                    actions_by_task[task_mode].append(action)


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
                rewards_by_task[task_mode].append(total_reward)
                len_by_task[task_mode].append(ep_steps)
                


    mean_return = float(np.mean(total_rewards))
    mean_episode_len = float(np.mean(episode_len))
    
    print(f"Eval Return: {mean_return:.2f}")
    writer.add_scalar("eval/episodic_return", mean_return, global_step)
    writer.add_scalar("eval/episodic_length", mean_episode_len, global_step)

    # Task specific:
    mean_return_task1 = float(np.mean(rewards_by_task[1.0]))
    mean_return_task2 = float(np.mean(rewards_by_task[-1.0]))
    writer.add_scalar("eval/episodic_return_task1", mean_return_task1, global_step)
    writer.add_scalar("eval/episodic_return_task2", mean_return_task2, global_step)
    writer.add_scalar("eval/episodic_length_task1", float(np.mean(len_by_task[1.0])), global_step)
    writer.add_scalar("eval/episodic_length_task2", float(np.mean(len_by_task[-1.0])), global_step)

    # Log Action Magnitude
    actions = np.stack(action_list, axis=0)  # [T, act_dim]
    mean_abs = float(np.mean(np.abs(actions)))
    max_abs  = float(np.max(np.abs(actions)))
    mean_raw = float(np.mean(actions))
    writer.add_scalar("eval/action_mean_abs", mean_abs, global_step)
    writer.add_scalar("eval/action_max_abs", max_abs, global_step)
    writer.add_scalar("eval/action_mean", mean_raw, global_step)



    # Average return over all evaluation episodes
    return mean_return, mean_episode_len

def evaluate_target_goal(args, actor, eval_env, writer, device, global_step):
    """
    Evaluation for task_mode = 'target_goal':
    - Starting goal: discrete points on circle (fixed positions)
    """

    goal_center = getattr(args, "goal_center", (0.0, 0.0))           
    goal_radius = float(getattr(args, "goal_radius", 25.0))            
    num_fixed_positions = int(getattr(args, "num_fixed_positions", 8))
    angles = np.linspace(0.0, 2.0*np.pi, num_fixed_positions, endpoint=False)

    with torch.no_grad():
        total_returns = []
        total_lengths = []
        action_list = []

     
        returns_by_initial_goal = {i: [] for i in range(num_fixed_positions)}

        for ep_idx in range(args.evaluate_episode_num):
            
            init_goal_idx = int(np.random.randint(0, num_fixed_positions))
            theta0 = angles[init_goal_idx]
            init_goal = (
                float(goal_center[0] + goal_radius * np.cos(theta0)),
                float(goal_center[1] + goal_radius * np.sin(theta0)),
            )

            options = {"target_goal": init_goal}
            state, info = eval_env.reset(options=options)

            ep_return = 0.0
            ep_steps = 0

            for s in range(args.evaluate_episode_maxstep):
                state_tensor = torch.as_tensor(state, dtype=torch.float32, device=device)
                action = actor.action(state_tensor, deterministic=True)

                if args.use_action_clipping:
                    action = np.clip(action, args.action_space_low, args.action_space_high)

                action_list.append(action)

                next_state, reward, terminated, truncated, info = eval_env.step(action)
                ep_return += reward
                ep_steps += 1

                done = terminated or truncated
                if done:
                    break
                state =  next_state

            total_returns.append(ep_return)
            total_lengths.append(ep_steps)
            returns_by_initial_goal[init_goal_idx].append(ep_return)

        mean_return = float(np.mean(total_returns))
        mean_len = float(np.mean(total_lengths))

    # ---- Logging ----
    writer.add_scalar("eval/episodic_return", mean_return, global_step)
    writer.add_scalar("eval/episodic_length", mean_len, global_step)

    # Per initial goal (diskrete Kreispositionen) loggen
    for i in range(num_fixed_positions):
        if len(returns_by_initial_goal[i]) > 0:
            writer.add_scalar(f"eval/return_init_goal_idx_{i}", float(np.mean(returns_by_initial_goal[i])), global_step)

    # Action stats korrekt
    actions = np.stack(action_list, axis=0)
    writer.add_scalar("eval/action_mean_abs", float(np.mean(np.abs(actions))), global_step)
    writer.add_scalar("eval/action_max_abs",  float(np.max(np.abs(actions))), global_step)
    writer.add_scalar("eval/action_mean",     float(np.mean(actions)), global_step)

    return mean_return, mean_len
