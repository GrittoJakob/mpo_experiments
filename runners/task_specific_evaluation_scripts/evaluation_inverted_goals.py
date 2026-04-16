import torch
import numpy as np
from writer.logging import logging


def evaluate_inverted_goal(args, actor, eval_env, writer, device, global_step):
    """
    Run evaluation episodes for both inverted-goal tasks
    and log everything via metrics dict.
    """

    with torch.no_grad():
        total_rewards = []
        episode_len = []

        rewards_by_task = {}
        len_by_task = {}
        task_reward_by_task = {}

        eval_tasks = [{"task_mode": 1.0}, {"task_mode": -1.0}]

        for task_options in eval_tasks:
            task_mode = task_options["task_mode"]

            rewards_by_task.setdefault(task_mode, [])
            task_reward_by_task.setdefault(task_mode, [])
            len_by_task.setdefault(task_mode, [])

            for _ in range(args.evaluate_episode_num):
                total_reward = 0.0
                total_task_reward = 0.0
                ep_steps = 0

                state, info = eval_env.reset(options=task_options)

                while True:
                    state_tensor = torch.as_tensor(state, dtype=torch.float32, device=device)
                    action = actor.action(state_tensor, deterministic=True)

                    next_state, reward, terminated, truncated, info = eval_env.step(action)
                    done = terminated or truncated
                    ep_steps += 1

                    task_reward = float(info.get("velocity_reward", 0.0))
                    total_task_reward += task_reward
                    total_reward += float(reward)

                    if done:
                        break
                    state = next_state

                total_rewards.append(total_reward)
                episode_len.append(ep_steps)

                rewards_by_task[task_mode].append(total_reward)
                len_by_task[task_mode].append(ep_steps)
                task_reward_by_task[task_mode].append(total_task_reward)

    mean_return = float(np.mean(total_rewards))
    mean_episode_len = float(np.mean(episode_len))

    metrics = {
        "eval/episodic_return": mean_return,
        "eval/episodic_length": mean_episode_len,
        "eval/episodic_return_task_forward": float(np.mean(rewards_by_task[1.0])),
        "eval/episodic_return_task_backward": float(np.mean(rewards_by_task[-1.0])),
        "eval/episodic_length_task_forward": float(np.mean(len_by_task[1.0])),
        "eval/episodic_length_task_backward": float(np.mean(len_by_task[-1.0])),
        "eval/episodic_task_return_task_forward": float(np.mean(task_reward_by_task[1.0])),
        "eval/episodic_task_return_task_backward": float(np.mean(task_reward_by_task[-1.0])),
    }

    logging(metrics, global_step, writer)

    print(f"Eval Return: {mean_return:.2f}")
