from helpers.env_creator import make_video_env
import torch
import os
import glob
import wandb
import options

def log_one_episode_video(args, actor, device, name_prefix, global_steps):

    venv = make_video_env(args, args.run_name, name_prefix)
    target_curriculum = [1.0, -1.0]
    try:
        for target in target_curriculum:
            if args.task_mode == "inverted":
                options={"task_mode": target} # set task mode for eval env forward/backward
            else:    
                options = None # set task mode for eval env forward/backward
            # eine Episode deterministic laufen lassen
            actor.eval()
            state, _ = venv.reset(options = options)
            done = False
            steps = 0
            total_reward = 0.0
            with torch.no_grad():
                while not done and steps < args.video_max_steps:
                    st = torch.as_tensor(state, dtype=torch.float32, device=device)
                    action = actor.action(st, deterministic=True)
                    state, reward, terminated, truncated, _ = venv.step(action)
                    done = terminated or truncated
                    total_reward += float(reward)
                    steps += 1
            actor.train()

            
    finally:
        venv.close()

    video_folder = os.path.join(args.video_dir, args.run_name)  # videos/<run_name>
    mp4s = sorted(glob.glob(os.path.join(video_folder, "*.mp4")), key=os.path.getmtime)

    if not mp4s:
        return
    latest = mp4s[-1]

    wandb.log(
        {
            "rollout/video": wandb.Video(latest, format="mp4"),
            "rollout/video_return": total_reward,
            "rollout/video_len": steps,
        },
        step=global_steps,
    )

    # optional: clean up
    os.remove(latest)