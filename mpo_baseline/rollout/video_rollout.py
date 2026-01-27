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

def _wait_for_mp4_complete(path, timeout_s=10.0, interval_s=0.2, stable_checks=3, min_bytes=50_000):
    """
    Wait until mp4 file exists and its size is stable for a few checks.
    Prevents logging half-written mp4s (common cause of broken/jumpy videos).
    """
    t0 = time.time()
    last_size = -1
    stable = 0

    while time.time() - t0 < timeout_s:
        if not os.path.exists(path):
            time.sleep(interval_s)
            continue

        size = os.path.getsize(path)
        if size >= min_bytes and size == last_size:
            stable += 1
            if stable >= stable_checks:
                return True
        else:
            stable = 0

        last_size = size
        time.sleep(interval_s)

    return False


def log_one_episode_video(args, actor, device, name_prefix, global_steps):
    video_folder = os.path.join(args.video_dir, args.run_name)
    os.makedirs(video_folder, exist_ok=True)

    # ---- pick exactly 2 tasks ----
    if args.task_mode == "inverted":
        task_options = [
            {"task_mode": 1.0},
            {"task_mode": -1.0},
        ]

    elif args.task_mode == "target_goal":
        R = float(getattr(args, "goal_radius", 5.0))
        goal_list = [
            ( R, 0.0), (-R, 0.0), (0.0,  R), (0.0, -R),
            ( R,  R), ( R, -R), (-R,  R), (-R, -R),
        ]
        idx = np.random.choice(len(goal_list), size=2, replace=(len(goal_list) < 2))
        g1, g2 = goal_list[int(idx[0])], goal_list[int(idx[1])]
        task_options = [
            {"target_goal": g1},
            {"target_goal": g2},
        ]

    else:
        task_options = [None, None]

    # ---- record two videos: video_1, video_2 ----
    for i in [1, 2]:
        prefix = f"video_{i}"  # fixed prefix, no fancy names
        env_options = task_options[i - 1]

        # Clean old files for this prefix (avoid picking wrong mp4)
        for f in glob.glob(os.path.join(video_folder, f"{prefix}*.mp4")):
            try:
                os.remove(f)
            except OSError:
                pass

        venv = make_video_env(args, args.run_name, prefix)

        try:
            actor.eval()
            state, _ = venv.reset(options=env_options)

            done, steps = False, 0
            with torch.no_grad():
                while not done and steps < args.video_max_steps:
                    st = torch.as_tensor(state, dtype=torch.float32, device=device)
                    action = actor.action(st, deterministic=True)
                    state, reward, terminated, truncated, _ = venv.step(action)
                    done = bool(terminated or truncated)
                    steps += 1

        finally:
            actor.train()
            venv.close()

        # Find the mp4 for this prefix
        mp4s = sorted(glob.glob(os.path.join(video_folder, f"{prefix}*.mp4")), key=os.path.getmtime)
        if not mp4s:
            print(f"[WARN] No mp4 found for {prefix} in {video_folder}")
            continue

        mp4 = mp4s[-1]

        # Wait until file is fully written (prevents broken videos)
        _wait_for_mp4_complete(mp4)

        wandb.log({f"rollout/{prefix}": wandb.Video(mp4, format="mp4")}, step=global_steps)
