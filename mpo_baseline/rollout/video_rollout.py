from helpers.env_creator import make_video_env
import torch
import os
import glob
import wandb
import numpy as np
import time
import math

def _extract_xy_from_info_or_env(info, env):
    """Gibt (x, y) zurück. Erst aus info, sonst aus Mujoco, sonst (nan, nan)."""
    x = info.get("x_position", None)
    y = info.get("y_position", None)

    if x is not None and y is not None:
        return float(x), float(y)

    # Manche Envs haben nur x_position
    if x is not None and y is None:
        return float(x), float("nan")

    # Fallback: Mujoco qpos (wenn vorhanden)
    try:
        data = env.unwrapped.data
        qpos = data.qpos
        # meist: qpos[0]=x, qpos[1]=y
        return float(qpos[0]), float(qpos[1]) if len(qpos) > 1 else float("nan")
    except Exception:
        return float("nan"), float("nan")

def log_one_episode_video(args, actor, device, name_prefix, global_steps):
    video_folder = os.path.join(args.video_dir, args.run_name)
    os.makedirs(video_folder, exist_ok=True)

    max_steps = getattr(args, "evaluate_episode_maxstep", getattr(args, "video_max_steps", 1000))

    # ---- pick exactly 2 tasks ----
    if args.task_mode in ("inverted", "inverted_multi_task"):
        task_options = [{"task_mode": 1.0}, {"task_mode": -1.0}]
        task_names   = ["forward", "backward"]
    else:
        task_options = [None]
        task_names   = ["task1"]

    best_reward = -float("inf")
    best_trajectory = None

    for i, (env_options, task_name) in enumerate(zip(task_options, task_names), start=1):
        prefix = f"{name_prefix}_video{i}" if name_prefix else f"video_{i}"
        before = set(glob.glob(os.path.join(video_folder, f"{prefix}*.mp4")))

        venv = make_video_env(args, args.run_name, prefix)
        total_reward, steps = 0.0, 0

        x_pos, y_pos = [], []
        x_goal, y_goal = [], []

        try:
            actor.eval()

            if env_options is None:
                state, info0 = venv.reset()
            else:
                state, info0 = venv.reset(options=env_options)

            done = False
            with torch.no_grad():
                while (not done) and (steps < max_steps):
                    st = torch.as_tensor(state, dtype=torch.float32, device=device)
                    action = actor.action(st, deterministic=True)

                    # robust: action -> numpy
                    if torch.is_tensor(action):
                        action_np = action.detach().cpu().numpy()
                    else:
                        action_np = np.asarray(action)

                    state, reward, terminated, truncated, info = venv.step(action_np)

                    # goals: nur wenn vorhanden, sonst NaN
                    x_goal.append(float(info.get("goal_x", float("nan"))))
                    y_goal.append(float(info.get("goal_y", float("nan"))))

                    # positions: robust extrahieren
                    x, y = _extract_xy_from_info_or_env(info, venv)
                    x_pos.append(x)
                    y_pos.append(y)

                    done = bool(terminated or truncated)
                    total_reward += float(reward)
                    steps += 1

        finally:
            actor.train()
            venv.close()

        # best trajectory über Tasks hinweg merken
        if total_reward > best_reward:
            best_reward = total_reward
            best_trajectory = list(zip(x_goal, y_goal, x_pos, y_pos))

        # Neue mp4 finden
        mp4 = None
        for _ in range(10):
            after = set(glob.glob(os.path.join(video_folder, f"{prefix}*.mp4")))
            new_files = list(after - before)
            if new_files:
                mp4 = max(new_files, key=os.path.getmtime)
                break
            time.sleep(0.1)

        if mp4 is None:
            mp4s = sorted(glob.glob(os.path.join(video_folder, f"{prefix}*.mp4")), key=os.path.getmtime)
            if not mp4s:
                print(f"[WARN] No mp4 found for prefix={prefix} in {video_folder}")
                continue
            mp4 = mp4s[-1]

        wandb.log(
            {
                f"rollout/video_{i}": wandb.Video(mp4, format="mp4"),
                f"rollout/video_{i}_return": total_reward,
                f"rollout/video_{i}_len": steps,
                f"rollout/video_{i}_task": task_name,
                # optional: wenn dein Wrapper es setzt
                f"rollout/video_{i}_task_direction": (1.0 if task_name == "forward" else -1.0),
            },
            step=global_steps,
        )

        try:
            os.remove(mp4)
        except OSError:
            pass

    return best_trajectory, best_reward
