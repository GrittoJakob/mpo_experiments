from dataclasses import dataclass
from typing import Optional

@dataclass
class BaseArgs:
    # ===============================
    # General System & Environment
    # ===============================

    device: str = "cuda"
    """device used for training ('cpu' or 'cuda')"""
    buffer_on_cuda: bool = True
    """store replay_buffer on cuda"""
    num_envs: int = 4
    """number of environments for rollout, currently used: AsyncVecEnv"""

    # Logging
    wandb_track: bool = True
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "MPO_Ant"
    """the wandb's project name"""
    wandb_entity: Optional[str] = "fsandco"
    """the entity (team) of wandb's project"""
    capture_video: bool = True
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    log_period: int = 90
    "number of global updates per log to wandb"
    video_dir: str = "videos"
    """where RecordVideo writes mp4s"""
    log_videos_period: int = 20
    """iterations per logging exactly one episode video"""
    exp_name: str = "mpo"
    """the name of this experiment"""
    log_dir: str = "mpo_logs"
    """directory used for logs and model checkpoints"""

    # ===============================
    # MPO Algorithm Parameters
    # ===============================

    # Actor & Critic
    std_init: float = 0.7
    """desired std for actor inialization on diagonal"""
    clip_to_env: bool = True
    """whether to use action clipping in rollout to save clipped action or not"""
    use_tanh_on_mean:bool = False
    """use tanh on mean in actor after last layer"""    
    target_update_period: int = 200
    "number of Q-updates steps per new target init"
    use_state_dependent_var: bool = True
    """whether std of actor is computed state-dependent oder independent"""
    

    # E-Step
    eta_lr : float = 1e-3
    "Learning rate for dual function"
    dual_constraint: float = 0.1
    """hard constraint of the dual formulation in the E-step"""
    use_action_penalty: bool = True
    """flag for using action penalty"""
    eps_penalty:float = 1e-3
    """constrain for action penalty loss term"""
    penalty_mix: float = 0.25
    """parameter for mixing dual_losses in E-Step (action penalty and normal dual loss)"""
    
    # M-Step
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
    use_mass_force_KL: bool = True
    """flag for using mass-forced KL divergence in the M-Step"""
    init_eta_mu: float = 0.5
    """int value of eta mu"""
    init_eta_sigma: float = 6
    """ init value of eta sigma"""
    init_eta_dual: float = 1.5
    """ init value of temperature variable"""

    # TD-Learning
    discount_factor: float = 0.99
    """discount factor gamma, used in TD updates for the critic"""
    q_loss_type: str = 'mse'
    """loss function type for the critic, e.g. 'mse' or 'huber'"""
    sample_action_num: int = 20
    """number of action samples per state for E-step weighting"""
    
    # Train Loop    

    warm_up_steps: int = 5000
    """number of warm-up steps for the buffer"""
    delay_policy_update: int = 2
    """number of critic updates per policy update"""
    sample_steps_per_iter: int = 1600
    """number of env steps per rollout in every iteration"""



    # ===============================
    # Evaluation Parameters
    # ===============================

    evaluate_period: int = 5
    """evaluate the agent every N iterations"""
    evaluate_episode_num: int = 5
    """how many evaluation episodes to run"""

    # ===============================
    # Logging / Checkpointing
    # ===============================

    load: Optional[str] = None
    """optional checkpoint file to load before training"""
    save_every_env_steps: int = 1000000
    """Flag how often actor and critic are saved in checkpoints"""

    # ===============================
    # warm up compilation
    # ===============================
    use_compile: bool = False
    """torch.compilation flag"""
    compile_mode: str=  "default"
    """compile mode for torch compilation"""

