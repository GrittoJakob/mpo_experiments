import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import wandb


def store_trajectory(
    trajectory,
    args=None,
    global_step=None,
    reward=None,
    name="best_traj",
    out_dir=None,
    log_distance_curve=True,
):
    """
    trajectory: iterable of (goal_x, goal_y, x_pos, y_pos)
                (kann auch zip(...) sein)
    Speichert Excel lokal + loggt Table/Plot in W&B (+ Artifact Upload).
    """

    # 1) Iterator -> Liste (zip ist sonst "one-shot")
    traj_list = list(trajectory)
    if len(traj_list) == 0:
        print("[WARN] store_trajectory: empty trajectory")
        return None

    # 2) DataFrame bauen
    arr = np.asarray(traj_list, dtype=np.float32)  # shape [T,4]
    df = pd.DataFrame(arr, columns=["goal_x", "goal_y", "x", "y"])
    df.insert(0, "t", np.arange(len(df), dtype=np.int32))

    # Distanz zum Goal
    df["dist_to_goal"] = np.sqrt((df["goal_x"] - df["x"])**2 + (df["goal_y"] - df["y"])**2)

    # Goal-Change Marker (falls Goal während Episode wechselt)
    gx_diff = df["goal_x"].diff().abs().fillna(0.0)
    gy_diff = df["goal_y"].diff().abs().fillna(0.0)
    df["goal_change"] = (gx_diff > 1e-6) | (gy_diff > 1e-6)
    df.loc[0, "goal_change"] = True

    # 3) Excel speichern
    if out_dir is None:
        # sinnvolle default location
        if args is not None:
            out_dir = os.path.join(args.video_dir, args.run_name, "trajectories")
        else:
            out_dir = os.path.join(".", "trajectories")
    os.makedirs(out_dir, exist_ok=True)

    step_str = f"_step{int(global_step)}" if global_step is not None else ""
    xlsx_path = os.path.join(out_dir, f"{name}{step_str}.xlsx")
    df.to_excel(xlsx_path, index=False, engine="openpyxl")

    # 4) W&B Logging (Table + Plot)
    log_dict = {}

    # Table
    try:
        table = wandb.Table(dataframe=df)
    except Exception:
        table = wandb.Table(columns=list(df.columns), data=df.values.tolist())
    log_dict[f"traj/{name}/table"] = table

    # 2D Plot (Ist vs Soll)
    goals = df.loc[df["goal_change"], ["goal_x", "goal_y"]].to_numpy(dtype=np.float32)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(df["x"].to_numpy(), df["y"].to_numpy(), linewidth=2, label="agent path (ist)")
    if goals.shape[0] > 0:
        ax.scatter(goals[:, 0], goals[:, 1], marker="x", s=80, label="goals (soll)")
        ax.plot(goals[:, 0], goals[:, 1], linestyle="--", linewidth=1, label="goal sequence")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    title = f"{name}"
    if reward is not None:
        title += f" | return={float(reward):.2f}"
    ax.set_title(title)
    ax.grid(True)
    ax.legend(loc="best")
    log_dict[f"traj/{name}/plot"] = wandb.Image(fig)
    plt.close(fig)

    # optional: Distanz-Kurve
    if log_distance_curve:
        fig2, ax2 = plt.subplots(figsize=(7, 3))
        ax2.plot(df["t"].to_numpy(), df["dist_to_goal"].to_numpy(), linewidth=2)
        ax2.set_xlabel("t")
        ax2.set_ylabel("dist_to_goal")
        ax2.set_title(f"{name} | distance to goal")
        ax2.grid(True)
        log_dict[f"traj/{name}/dist_curve"] = wandb.Image(fig2)
        plt.close(fig2)

    # Scalars
    if reward is not None:
        log_dict[f"traj/{name}/return"] = float(reward)
    log_dict[f"traj/{name}/len"] = int(len(df))
    log_dict[f"traj/{name}/xlsx_saved"] = xlsx_path  # nur als String-Info

    wandb.log(log_dict, step=global_step)

    # 5) Excel zusätzlich als Artifact hochladen (damit in W&B downloadbar)
    try:
        art_name = f"{name}{step_str}".replace("/", "_")
        artifact = wandb.Artifact(name=art_name, type="trajectory")
        artifact.add_file(xlsx_path)
        wandb.log_artifact(artifact)
    except Exception as e:
        print(f"[WARN] Could not log artifact: {e}")

    print(f"[OK] Trajectory saved to: {xlsx_path}")
    return xlsx_path
