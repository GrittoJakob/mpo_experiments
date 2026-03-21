import torch
import numpy as np

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

            while True:
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
