from helpers.env_creator import make_video_env
import torch
import os
import glob
import wandb
import options
import numpy as np
from helpers.env_creator import make_video_env
import torch
import os
import glob
import wandb
import numpy as np
import time
import os, glob, time
import numpy as np
import torch
import wandb


def log_one_episode_video(args, actor, device, name_prefix, global_steps):
    video_folder = os.path.join(args.video_dir, args.run_name)
    os.makedirs(video_folder, exist_ok=True)

    # max steps (wie früher)
    max_steps = getattr(args, "evaluate_episode_maxstep", getattr(args, "video_max_steps", 1000))

    # ---- pick exactly 2 tasks ----
    if args.task_mode == "inverted":
        task_options = [{"task_mode": 1.0}, {"task_mode": -1.0}]
        task_names   = ["forward", "backward"]

    else:
        task_options = [None, None]
        task_names   = ["task1", "task2"]

    for i, (env_options, task_name) in enumerate(zip(task_options, task_names), start=1):
        prefix = f"{name_prefix}_video{i}" if name_prefix else f"video_{i}"

        # Snapshot vorheriger Dateien (kein löschen nötig)
        before = set(glob.glob(os.path.join(video_folder, f"{prefix}*.mp4")))

        venv = make_video_env(args, args.run_name, prefix)
        total_reward, best_reward, steps = 0.0, 0.0,  0


        x_pos = []
        y_pos = []
        x_goal = []
        y_goal = []

        try:
            actor.eval()
            if env_options is None:
                state, _ = venv.reset()
            else:
                state, _ = venv.reset(options=env_options)

            done = False
            with torch.no_grad():
                while (not done) and (steps < max_steps):
                    st = torch.as_tensor(state, dtype=torch.float32, device=device)
                    action = actor.action(st, deterministic=True)
                    state, reward, terminated, truncated, info = venv.step(action)

                    x_goal.append(info["goal_x"])
                    y_goal.append(info["goal_y"])
                    x_pos.append(info["x_position"])
                    y_pos.append(info["y_position"])

                    done = bool(terminated or truncated)
                    total_reward += float(reward)
                    steps += 1
        finally:
            if total_reward > best_reward:
                best_reward = total_reward
                trajectory = zip(x_goal, y_goal, x_pos, y_pos)
            actor.train()
            venv.close()

        # Neue mp4 finden (kurzer, schneller wait bis sie da ist)
        mp4 = None
        for _ in range(10):  # ~1s max
            after = set(glob.glob(os.path.join(video_folder, f"{prefix}*.mp4")))
            new_files = list(after - before)
            if new_files:
                mp4 = max(new_files, key=os.path.getmtime)
                break
            time.sleep(0.1)

        # Fallback: nimm die neueste mit Prefix
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
            },
            step=global_steps,
        )

        # optional cleanup
        try:
            os.remove(mp4)
        except OSError:
            pass
    
    return trajectory, best_reward
