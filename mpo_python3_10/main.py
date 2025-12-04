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
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    log_inner_interval: int = 25
    "number of global updates per log to wandb"
    num_threads: int = 16  
    """number of threads to use"""

    # ===============================
    # MPO Algorithm Parameters
    # ===============================
    iteration_num: int = 1000
    """number of outer MPO iterations"""
    sample_action_num: int = 20
    """number of action samples per state for E-step weighting"""
    target_update_period: int = 200
    "number of Q-updates steps per new target init"
    log_dir: str = "mpo_logs"
    """directory used for logs and model checkpoints"""
    discount_factor: float = 0.99
    """discount factor gamma, used in TD updates for the critic"""
    hidden_size_actor: int = 256
    """hidden size of actor network"""
    hidden_size_critic: int = 512
    """hidden size of critc network"""
    use_retrace: bool = False
    """True for use of retrace, false for TD approach"""

    mstep_iteration_num: int = 5
    """number of gradient updates in the M-step"""
    learning_rate: float = 1e-4
    """Learning rate for Adam optimizer"""
    eta_lr : float = 1e-2
    "Learning rate for dual function"
    dual_constraint: float = 0.1
    """hard constraint of the dual formulation in the E-step"""
    kl_mean_constraint: float = 0.005   
    """hard constraint of the mean in the M-step"""
    kl_var_constraint: float = 0.0001        
    """hard constraint of the covariance in the M-step"""
    alpha_mean_scale: float = 1.0
    """learning rate / scale factor for updating eta_mu (mean KL Lagrange multiplier)"""
    alpha_var_scale: float = 0.1
    """learning rate / scale factor for updating eta_sigma (variance KL Lagrange multiplier)"""
    alpha_mean_max: float = 0.1
    """maximum clamp value for eta_mu (mean KL Lagrange multiplier)"""
    alpha_var_max: float = 1.0
    """maximum clamp value for eta_sigma (variance KL Lagrange multiplier)"""
    q_loss_type: str = 'mse'
    """loss function type for the critic, e.g. 'mse' or 'huber'"""
    UTD_ratio: float = 1.0
    """ Ratio: num_updates per env step"""
    max_replay_buffer: int = 2000000
    """maximum number of transitions stored; FIFO removes oldest episodes when exceeded"""
    std_init: float = 0.7
    """desired std for actor inialization on diagonal"""
    warm_up_steps: int = 10000
    """number of warm-up steps for the buffer"""
    delay_policy_update: int = 1
    """number of critic updates per policy update"""

    # ===============================
    # Sampling / Replay Buffer
    # ===============================
    sample_episode_num: int = 1
    """number of episodes sampled per MPO iteration"""
    sample_episode_maxstep: int = 1000
    """maximum number of steps per sampled episode"""
    batch_size: int = 512
    """batch size used when sampling from replay buffer"""
    print_replay_buffer: bool = False
    """Print shape and one episode from replay buffer for debugging"""

    # ===============================
    # Evaluation Parameters
    # ===============================

    evaluate_period: int = 1
    """evaluate the agent every N iterations"""
    evaluate_episode_num: int = 10
    """how many evaluation episodes to run"""
    evaluate_episode_maxstep: int = 1000
    """max steps per evaluation episode"""

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
        
def limit_threads(n: int):
    # PyTorch threads
    torch.set_num_threads(n)
    torch.set_num_interop_threads(1)

    # NumPy / BLAS threads (muss vor Imports passieren, aber auch so meist ok)
    os.environ["OMP_NUM_THREADS"] = str(n)
    os.environ["OPENBLAS_NUM_THREADS"] = str(n)
    os.environ["MKL_NUM_THREADS"] = str(n)
    os.environ["NUMEXPR_NUM_THREADS"] = str(n)
   

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

    global_update = logs["global_update"]

    # --- TensorBoard ---
    for key in [
        "iteration",
        "num_steps",
        "global_update",
        "mean_return_buffer",
        "mean_reward_buffer",
        "mean_loss_q",
        "mean_loss_p",
        "mean_loss_l",
        "mean_current_q",
        "mean_target_q",
        "eta",
        "runtime_env",
        "runtime_eval",
        "runtime_policy_eval",
        "runtime_M_step",
        "runtime_E_step",
        "max_kl_sigma",
        "mean_sigma_det",
        "eta_mu",
        "eta_sigma",
        "return_eval",
        "var_mean",
        "var_min",
        "var_max"
    ]:
        if key in logs:
            tag = key.replace("_", "/") if key.startswith("eval_") else key
            writer.add_scalar(tag, logs[key], global_update)

    # --- W&B ---
    wandb.log(logs, step=global_update)
    
   
if __name__ == "__main__":
    args = tyro.cli(Args)  # CLI aus der Dataclass
    limit_threads(args.num_threads)

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

     # Device
    device = torch.device("cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")
    args.device = str(device)  # sicherstellen, dass MPO das gleiche sieht

    # Env erstellen (ein einzelnes Env für MPO)
    env = make_env(args.env_id, args.capture_video, run_name)
    assert isinstance(env.action_space, gym.spaces.Box)
    

    # Seeding 
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False    
    env.reset(seed=args.seed)  


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

    writer.close()
    if args.track:
        wandb.finish()
    env.close()

