import torch
import numpy as np
try:
    import wandb
except ImportError:
    wandb = None
from runners.task_specific_evaluation_scripts.evaluation_inverted_goals import evaluate_inverted_goal
from runners.task_specific_evaluation_scripts.evaluation_ERFI_noise import evaluate_erfi
from runners.task_specific_evaluation_scripts.evaluation_target_goals import evaluate_target_goal
from writer.logging import _wandb_is_active, _to_python_number

def evaluate(args, actor, eval_env, writer, device, grad_updates):
    """
    Run evaluation episodes using the current policy (self.actor)
    and return the average total reward per episode.
    """
    task_mode = getattr(args, "task_mode", "default")
    rand_mode = getattr(args, "rand_mode", "default")

    if task_mode in ["inverted_without_task_hint"]:
        return evaluate_inverted_goal(args, actor, eval_env, writer, device, grad_updates)
    elif task_mode in ("target_goal"):
        return evaluate_target_goal(args, actor, eval_env, writer, device, grad_updates)
    elif rand_mode == "ERFI":
        return evaluate_erfi(args, actor, eval_env, writer, device, grad_updates)

    
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
                state, _ = eval_env.reset()

                # Loop over one episode
                while True:

                    # Convert state to tensor on the correct device
                    state_tensor = torch.as_tensor(
                        state, dtype=torch.float32, device=device
                        )

                    # Get action from actor
                    action = actor.action(state_tensor, deterministic = True)   # Already numpy 
                    action_list.append(action)

                    # Environment step
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
    metrics = {
        "eval/episodic_return": mean_return,
        "eval/episodic_length": mean_episode_len,
    }
    
    if writer is not None:
        for key, value in metrics.items():
            if key == "grad_updates":
                continue
            writer.add_scalar(key, _to_python_number(value), grad_updates)
        writer.flush()

    if _wandb_is_active():
        wandb.log(
            {key: _to_python_number(value) for key, value in metrics.items()},
            step=grad_updates,
        )
