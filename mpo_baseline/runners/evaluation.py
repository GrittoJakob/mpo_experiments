import gymnasium as gym
import torch
import numpy as np
import options


def evaluate(args, actor, eval_env, writer, device, global_step):
    """
    Run evaluation episodes using the current policy (self.actor)
    and return the average total reward per episode.
    """

    if args.task_mode in ["inverted_without_task_hint"]:
        return evaluate_inverted(args, actor, eval_env, writer, device, global_step)
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
    

def evaluate_inverted(args, actor, eval_env, writer, device, global_step):
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

    # Per initial goal 
    for i in range(num_fixed_positions):
        if len(returns_by_initial_goal[i]) > 0:
            writer.add_scalar(f"eval/return_init_goal_idx_{i}", float(np.mean(returns_by_initial_goal[i])), global_step)

    return mean_return, mean_len


def evaluate_erfi(args, actor, eval_env, writer, device, global_step):
    """
    ERFI evaluation for SINGLE task:
      - runs eval episodes with erfi_mode="RFI" and erfi_mode="RAO"
      - logs both variants separately
    """

    noise_modes = ["RFI", "RAO"]

    returns_by_mode = {m: [] for m in noise_modes}
    lens_by_mode    = {m: [] for m in noise_modes}
    taskret_by_mode = {m: [] for m in noise_modes}  # e.g. sum velocity_reward (optional)

    actions_by_mode = {m: [] for m in noise_modes}

    actor.eval()
    with torch.no_grad():
        for mode in noise_modes:
            for ep_idx in range(args.evaluate_episode_num):
                total_reward = 0.0
                total_task_reward = 0.0
                ep_steps = 0

                state, info = eval_env.reset(options={"erfi_mode": mode})

                for _ in range(args.evaluate_episode_maxstep):
                    state_tensor = torch.as_tensor(state, dtype=torch.float32, device=device)

                    action = actor.action(state_tensor, deterministic=True)
                    if args.use_action_clipping:
                        action = np.clip(action, args.action_space_low, args.action_space_high)

                    actions_by_mode[mode].append(action)

                    next_state, reward, terminated, truncated, info = eval_env.step(action)
                    done = bool(terminated) or bool(truncated)

                    ep_steps += 1
                    total_reward += float(reward)

                    # optional: if present in info (otherwise 0.0)
                    total_task_reward += float(info.get("velocity_reward", 0.0))

                    if done:
                        break
                    state = next_state

                returns_by_mode[mode].append(total_reward)
                lens_by_mode[mode].append(ep_steps)
                taskret_by_mode[mode].append(total_task_reward)

    # ----- aggregate overall (over both modes) -----
    all_returns = returns_by_mode["RFI"] + returns_by_mode["RAO"]
    all_lens    = lens_by_mode["RFI"] + lens_by_mode["RAO"]

    mean_return = float(np.mean(all_returns)) if all_returns else 0.0
    mean_len    = float(np.mean(all_lens)) if all_lens else 0.0

    writer.add_scalar("eval/erfi/episodic_return", mean_return, global_step)
    writer.add_scalar("eval/erfi/episodic_length", mean_len, global_step)

    # ----- per mode logging -----
    for mode in noise_modes:
        r = returns_by_mode[mode]
        l = lens_by_mode[mode]
        tr = taskret_by_mode[mode]

        if r:
            writer.add_scalar(f"eval/erfi/episodic_return_{mode}", float(np.mean(r)), global_step)
            writer.add_scalar(f"eval/erfi/episodic_length_{mode}", float(np.mean(l)), global_step)
            writer.add_scalar(f"eval/erfi/episodic_task_return_{mode}", float(np.mean(tr)), global_step)

    print(
        f"ERFI Eval | mean_return(all): {mean_return:.2f}, mean_len(all): {mean_len:.1f} | "
        f"RFI: {np.mean(returns_by_mode['RFI']):.2f} | RAO: {np.mean(returns_by_mode['RAO']):.2f}"
    )
    actor.train()

    return mean_return, mean_len
