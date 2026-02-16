from dataclasses import dataclass
import os
import time
from typing import Optional

@dataclass
class Args:

    # ===============================
    # General System & Environment
    # ===============================
    exp_name: str = "mpo"
    """the name of this experiment"""
    device: str = "cuda"
    """device used for training ('cpu' or 'cuda')"""
    env_id: str = "Ant-v5"
    """gym environment name (used in gym.make)"""
    seed: int = 1
    """seed of the experiment"""
    wandb_track: bool = True
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "MPO_Ant"
    """the wandb's project name"""
    wandb_entity: Optional[str] = "fsandco"
    """the entity (team) of wandb's project"""
    capture_video: bool = True
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    log_period: int = 400
    "number of global updates per log to wandb"
    num_threads: int = 16
    """number of threads to use"""
    video_dir: str = "videos"
    """where RecordVideo writes mp4s"""
    log_videos_period: int = 75
    """iterations per logging exactly one episode video"""
    buffer_on_cuda: bool = True
    """store replay_buffer on cuda"""
    num_envs: int = 4
    asynchronous: bool = True
    ckpt_path = "/home/jakob_gritto/tum-adlr-ws26-09/mpo_baseline/checkpoints/Final_run_target_goal_for_poster.pt"
    

    # ===============================
    # MPO Algorithm Parameters
    # ===============================
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
    actor_lr: float = 1e-4
    """Learning rate for actor Adam optimizer"""
    critic_lr: float = 1e-4
    """Learning rate for critic adam optimizer"""
    eta_lr : float = 1e-3
    "Learning rate for dual function"
    dual_constraint: float = 0.1
    """hard constraint of the dual formulation in the E-step"""
    kl_mean_constraint: float = 1e-4   
    """hard constraint of the mean in the M-step"""
    kl_var_constraint: float = 1e-6        
    """hard constraint of the covariance in the M-step"""
    alpha_mean_scale: float = 1.0
    """learning rate / scale factor for updating eta_mu (mean KL Lagrange multiplier)"""
    alpha_var_scale: float = 0.5
    """learning rate / scale factor for updating eta_sigma (variance KL Lagrange multiplier)"""
    alpha_mean_max: float = 5.0
    """maximum clamp value for eta_mu (mean KL Lagrange multiplier)"""
    alpha_var_max: float = 100.00
    """maximum clamp value for eta_sigma (variance KL Lagrange multiplier)"""
    q_loss_type: str = 'mse'
    """loss function type for the critic, e.g. 'mse' or 'huber'"""
    UTD_ratio: float = 0.5
    """ Ratio: num_updates per env step"""
    max_replay_buffer: int = 800000
    """maximum number of transitions stored; FIFO removes oldest episodes when exceeded"""
    std_init: float = 0.7
    """desired std for actor inialization on diagonal"""
    warm_up_steps: int = 15000
    """number of warm-up steps for the buffer"""
    delay_policy_update: int = 2
    """number of critic updates per policy update"""
    init_eta_mu: float = 0.5
    """int value of eta mu"""
    init_eta_sigma: float = 6
    """ init value of eta sigma"""
    init_eta_dual: float = 1.5
    """ init value of temperature variable"""
    use_state_dependent_var: bool = True
    """whether std of actor is computed state-dependent oder independent"""
    use_action_penalty: bool = True
    """flag for using action penalty"""
    eps_penalty:float = 1e-3
    """constrain for action penalty loss term"""
    penalty_mix: float = 0.25
    """parameter for mixing dual_losses in E-Step (action penalty and normal dual loss)"""
    use_action_clipping: bool = True
    """whether to use action clipping in rollout to save clipped action or not"""
    use_tanh_on_mean:bool = True
    """use tanh on mean in actor after last layer"""
    use_mass_force_KL: bool = True

    # ===============================
    # Sampling / Replay Buffer
    # ===============================
    min_steps_per_env: int = 1000
    """number of env steps to sample per iteration"""
    sample_episode_maxstep: int = 1000
    """maximum number of steps per sampled episode"""
    batch_size: int = 1024
    """batch size used when sampling from replay buffer"""
    print_replay_buffer: bool = False
    """Print shape and one episode from replay buffer for debugging"""
    max_training_steps: int = 5000000
    """Maximal number of env steps for training"""

    # ===============================
    # Evaluation Parameters
    # ===============================

    evaluate_period: int = 5
    """evaluate the agent every N iterations"""
    evaluate_episode_num: int = 4
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

    # ===============================
    # warm up compilation
    # ===============================
    use_compile: bool = True
    """torch.compilation flag"""
    compile_mode: str=  "default"
    """compile mode for torch compilation"""

    # ===============================
    # Reward shaping for ant env
    # ===============================
    # ctrl_cost_weight: float = 0.5
    # """ Weight for ctrl_cost term, default = 0.5"""
    # healthy_reward_weight: float = 0.8
    # """Weight for healthy_reward term, default = 1.2, 0.8 for multi task"""
    # contact_cost_weight: float = 5e-4
    # """Weight for contact_cost term, default = 5e-4"""
    # forward_reward_weight: float = 1.0
    # """Weight for forward_reward term, default = 1"""


    # ===============================
    # Multi-Task settings
    # ===============================

    # Task Constraints
    ctrl_cost_weight: float = 0.5
    """ Weight for ctrl_cost term, default = 0.5"""
    healthy_reward_weight: float = 1.0
    """Weight for healthy_reward term, default = 1.2"""
    contact_cost_weight: float = 5e-4
    """Weight for contact_cost term, default = 5e-4"""
    forward_reward_weight: float = 1.0
    """Weight for forward_reward term, default = 1"""

    velocity_reward_scale: float = 2.0
    """scale parameter for reward flipped velocity"""
    scale_wrong_direction_reward: float = 1.0
    """scale parameter for scale if velocity is wrong direction"""
    movement_bonus_scale: float = 0.1
    """scale parameter for movement reward, movement in any direction"""
    tilt_penalty_weight: float = 1.0 
    """weight parameter for tilt penalty, penalizes only excess tilt over threshold"""
    death_penalty: float = -50.0
    position_reward_scale: float = 5
    success_radius: float = 2
    maximum_area: float = 100


    # ## in train_args.py 
    task_mode: str = "target_goal_ferdinand" 
    """Options: 'default' (Run Forward), 'velocity' (Match Speed), 'target' (Go to XY)"""
    # velocity_reward_scale: float = 1.75
    # """scale parameter for reward flipped velocity"""
    # scale_wrong_direction_reward: float = 1.75
    # """scale parameter for scale if velocity is wrong direction"""
    # position_reward_scale: float = 5
    # goal_radius: float = 25
    # success_radius: float = 2
    # maximum_area: float = 100
    # history_len: int = 5
    # """lenght of history to append in obs"""
    # append_task_reward: bool = False
    # """to append the specific task reward in obs"""
    track_trajectory: bool = False
    include_cfrc_ext_in_observation: bool = True
    """flag for exlude/include central force terms in observations"""

    
    rand_mode: str = "ERFI"
    """the environment parametrization type for meta-learning""" # Currently supports ERFI, RAO, RFI, and None (no param randomization)
    rand_split_ratio: float = 0.5
    """the ratio at which to split the population for ERFI mode, if 0.9 the first 90% of the population uses RFI and the last 10% uses RAO"""
    noise_limit: float = 0.05
    """noise limit for RFI and RAO"""

           

