import os
import time
import random
from dataclasses import dataclass
import numpy as np
import torch
import gymnasium as gym
from torch.utils.tensorboard import SummaryWriter
import tyro
import wandb
from typing import Optional

from mpo import MPO

@dataclass
class Args:

    # ===============================
    # General System & Environment
    # ===============================
    exp_name: str = "mpo_ant"
    """the name of this experiment"""
    device: str = "cpu"
    """device used for training ('cpu' or 'cuda')"""
    env_id: str = "Ant-v5"
    """gym environment name (used in gym.make)"""
    seed: int = 1
    """seed of the experiment"""
    track: bool = True
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "MPO_Ant"
    """the wandb's project name"""
    wandb_entity: Optional[str] = "adl-robotics-project"
    """the entity (team) of wandb's project"""
    capture_video: bool = True
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # ===============================
    # MPO Algorithm Parameters
    # ===============================
    iteration_num: int = 1000
    """number of outer MPO iterations"""
    log_dir: str = "mpo_logs"
    """directory used for logs and model checkpoints"""
    discount_factor: float = 0.99
    """discount factor gamma, used in TD updates for the critic"""

    mstep_iteration_num: int = 5
    """number of gradient updates in the M-step"""
    learning_rate: float = 3e-4
    """Learning rate for Adam optimizer"""
    dual_constraint: float = 0.1
    """hard constraint of the dual formulation in the E-step"""
    kl_mean_constraint: float = 0.0005   
    """hard constraint of the mean in the M-step"""
    kl_var_constraint: float = 0.00001        
    """hard constraint of the covariance in the M-step"""
    alpha_mean_scale: float = 1.0
    """learning rate / scale factor for updating eta_mu (mean KL Lagrange multiplier)"""
    alpha_var_scale: float = 100.0
    """learning rate / scale factor for updating eta_sigma (variance KL Lagrange multiplier)"""
    alpha_scale: float = 10.0
    """generic scale factor for joint KL, used in some MPO variants (optional / fallback)"""
    alpha_mean_max: float = 0.1
    """maximum clamp value for eta_mu (mean KL Lagrange multiplier)"""
    alpha_var_max: float = 10.0
    """maximum clamp value for eta_sigma (variance KL Lagrange multiplier)"""
    alpha_max: float = 10.0
    """maximum clamp value for generic dual variables (if used together with alpha_scale)"""
    q_loss_type: str = 'mse'
    """loss function type for the critic, e.g. 'mse' or 'huber'"""
    clear_replay_buffer: bool = False
    """if True: empty the replay buffer at every outer iteration (forces on-policy MPO behavior)"""
    max_replay_buffer: int = 2000000
    """maximum number of transitions stored; FIFO removes oldest episodes when exceeded"""

    # ===============================
    # Sampling / Replay Buffer
    # ===============================

    num_updates_per_iter: int = 5
    """how many passes over replay buffer per MPO iteration"""
    sample_episode_num: int = 50
    """number of episodes sampled per MPO iteration"""
    sample_episode_maxstep: int = 1000
    """maximum number of steps per sampled episode"""
    sample_action_num: int = 64
    """number of action samples per state for E-step weighting"""
    batch_size: int = 128
    """batch size used when sampling from replay buffer"""
    print_replay_buffer: bool = False
    """Print shape and one episode from replay buffer for debugging"""
    

    # ===============================
    # Evaluation Parameters
    # ===============================

    evaluate_period: int = 10
    """evaluate the agent every N iterations"""
    evaluate_episode_num: int = 1
    """how many evaluation episodes to run"""
    evaluate_episode_maxstep: int = 300
    """max steps per evaluation episode"""
    target_update_period: int = 250
    "number of Q-updates steps per new target init"

    # ===============================
    # Logging / Checkpointing
    # ===============================

    render: bool = False
    """render environment during sampling"""
    load: Optional[str] = None
    """optional checkpoint file to load before training"""
    save_every: int = 100
    """save full model every N MPO iterations"""
    save_latest: bool = True
    """always update a lightweight 'latest' checkpoint (fast, no replay buffer)"""
    save_replay_buffer: bool = True
    """whether to include replay buffer in checkpoints (large files!)"""
        
   
def make_env(env_id, capture_video, run_name):
    if capture_video:
        env = gym.make(env_id, render_mode="rgb_array")
        env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
    else:
        env = gym.make(env_id)

    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = gym.wrappers.ClipAction(env)
    return env

def log_callback(logs):
    it = logs["global_update"]

    # TensorBoard
    writer.add_scalar("mean_return_buffer", logs["mean_return_buffer"], it)
    writer.add_scalar("mean_reward_buffer", logs["mean_reward_buffer"], it)
    writer.add_scalar("loss_q",      logs["mean_loss_q"], it)
    writer.add_scalar("loss_p",      logs["mean_loss_p"], it)
    writer.add_scalar("loss_l",      logs["mean_loss_l"], it)
    writer.add_scalar("mean_q",      logs["mean_q"], it)
    writer.add_scalar("eta",         logs["eta"], it)
    writer.add_scalar("max_kl_mu",   logs["max_kl_mu"], it)
    writer.add_scalar("max_kl_sigma",logs["max_kl_sigma"], it)
    writer.add_scalar("mean_sigma_det", logs["mean_sigma_det"], it)
    writer.add_scalar("eta_mu",      logs["eta_mu"], it)
    writer.add_scalar("eta_sigma",   logs["eta_sigma"], it)

    if "return_eval" in logs:
        writer.add_scalar("eval/return_eval",      logs["return_eval"], it)
        writer.add_scalar("eval/max_return_eval",  logs["max_return_eval"], it)

    # W&B
    wandb.log(logs, step=it)
    
   
if __name__ == "__main__":
    args = tyro.cli(Args)  # CLI aus der Dataclass

    # Run-Name à la CleanRL
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

    # Logging-Verzeichnis
    os.makedirs(args.log_dir, exist_ok=True)
    tb_dir = os.path.join(args.log_dir, "tb", run_name)
    writer = SummaryWriter(tb_dir)

    # Hyperparameter-Text in TensorBoard
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n" + "\n".join(
            [f"|{key}|{value}|" for key, value in vars(args).items()]
        ),
    )

    # Seeding (wie CleanRL)
    #random.seed(args.seed)
    #np.random.seed(args.seed)
    #torch.manual_seed(args.seed)
    #torch.backends.cudnn.deterministic = True

    # Device
    device = torch.device("cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")
    args.device = str(device)  # sicherstellen, dass MPO das gleiche sieht

    # W&B
    if args.track:
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=False,   # TensorBoard -> W&B
            config=vars(args),
            name=run_name,
            monitor_gym=False,
            save_code=True,
        )

    # Env erstellen (ein einzelnes Env für MPO)
    env = make_env(args.env_id, args.capture_video, run_name)
    assert isinstance(env.action_space, gym.spaces.Box)
    #print(env.action_space.low, env.action_space.high)
    # MPO initialisieren
    Agent = MPO(env, args)   

    if args.load is not None:
        Agent.load_model(args.load)


    all_logs = Agent.train(
        iteration_num=args.iteration_num,
        render=args.render,
        log_callback = log_callback
    )
    
    if args.print_replay_buffer:
            Agent.replaybuffer.debug_summary()
            Agent.replaybuffer.print_episode(3,20)

    # Logs aus MPO in TB + W&B schreiben
    for logs in all_logs:
        it = logs["iteration"]

        # Konsole (optional)
        print(f"iteration: {it}")
        print(f"  mean_return   : {logs['mean_return']:.3f}")
        print(f"  mean_reward   : {logs['mean_reward']:.3f}")
        print(f"  mean_loss_q        : {logs['mean_loss_q']:.3f}")
        print(f"  mean_loss_p        : {logs['mean_loss_p']:.3f}")
        print(f"  mean_loss_l        : {logs['mean_loss_l']:.3f}")
        if "return_eval" in logs:
            print(f"  return_eval      : {logs['return_eval']:.3f}")
            print(f"  max_return_eval  : {logs['max_return_eval']:.3f}")


    writer.close()
    if args.track:
        wandb.finish()
    env.close()

