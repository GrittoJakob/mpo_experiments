import torch
import numpy as np

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
