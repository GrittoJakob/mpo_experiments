from environment.base_env_creator import make_video_env
import torch
import os
import glob
import wandb
import time




def log_one_episode_video(args, actor, device, name_prefix, grad_updates):
    
    # Video folder
    video_folder = os.path.join(args.video_dir, args.run_name)
    os.makedirs(video_folder, exist_ok=True)

    # Task modes
    task_mode = getattr(args, "task_mode", "default")
    if task_mode in ("inverted", "inverted_without_task_hint"):
        task_options = [{"task_mode": 1.0}, {"task_mode": -1.0}]
        task_names   = ["forward", "backward"]
    else:
        task_options = [None]
        task_names   = ["task1"]

    for i, (env_options, task_name) in enumerate(zip(task_options, task_names), start=1):
        prefix = f"{name_prefix}_video{i}" if name_prefix else f"video_{i}"
        before = set(glob.glob(os.path.join(video_folder, f"{prefix}*.mp4")))

        venv = make_video_env(args, args.run_name, prefix)
        total_reward, steps = 0.0, 0

        try:
            actor.eval()
            done = False
            if env_options is None:
                state, _ = venv.reset()
            else:
                state, _ = venv.reset(options=env_options)

            with torch.no_grad():
                while True:
                    states = torch.as_tensor(state, dtype=torch.float32, device=device)
                    action = actor.action(states, deterministic=True)
                    state, reward, terminated, truncated, info = venv.step(action)

                    total_reward += reward
                    steps += 1 

                    done = bool(terminated or truncated)
        
                    if done: 
                        break
                    
        finally:
            actor.train()
            venv.close()

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

        if args.wandb_track:
            wandb.log(
                {
                    f"rollout/video_{i}": wandb.Video(mp4, format="mp4"),
                    f"rollout/video_{i}_return": total_reward,
                    f"rollout/video_{i}_len": steps,
                    f"rollout/video_{i}_task": task_name,
                    f"rollout/video_{i}_task_direction": (1.0 if task_name == "forward" else -1.0),
                },
                step=grad_updates,
            )

        try:
            os.remove(mp4)
        except OSError:
            pass

    return
