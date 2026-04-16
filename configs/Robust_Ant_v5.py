from dataclasses import dataclass
from typing import Optional

@dataclass
class Robust_Ant_Args():

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
    wandb_project_name: str = "MPO_TRL"
    """the wandb's project name"""
    wandb_entity: Optional[str] = "jakobs_experiments"
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    log_period: int = 250
    "number of global updates per log to wandb"
    video_dir: str = "videos"
    """where RecordVideo writes mp4s"""
    log_videos_period: int = 20
    """iterations per logging exactly one episode video"""
    exp_name: str = "mpo"
    """the name of this experiment"""
    log_dir: str = "mpo_logs"
    """directory used for logs and model checkpoints"""

    # Environment
    env_id: str = "Ant-v5"
    """gym environment name (used in gym.make)"""
    seed: int = 1
    """seed of the experiment"""

    # ===============================
    # MPO Algorithm Parameters
    # ===============================


    # Actor & Critic
    hidden_size_actor: int = 128
    """hidden size of actor network"""
    hidden_size_critic: int = 128
    """hidden size of critc network"""
    actor_lr: float = 3e-4
    """Learning rate for actor Adam optimizer"""
    critic_lr: float = 3e-4
    """Learning rate for critic adam optimizer"""   
    std_init: float = 0.7
    """desired std for actor inialization on diagonal"""
    clip_to_env: bool = True
    """whether to use action clipping in rollout to save clipped action or not"""
    use_tanh_on_mean:bool = True
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
    use_action_penalty: bool = False
    """flag for using action penalty"""
    eps_penalty:float = 1e-3
    """constrain for action penalty loss term"""
    penalty_mix: float = 0.3
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
    UTD_ratio: float = 0.5
    """ Ratio: num_updates per env step"""
    max_training_steps: int = 20000
    """Maximal number of env steps for training"""
    warm_up_steps: int = 5000
    """number of warm-up steps for the buffer"""
    delay_policy_update: int = 2
    """number of critic updates per policy update"""
    sample_steps_per_iter: int = 4000
    """number of env steps per rollout in every iteration"""
   
    # Sampling / Replay Buffer
    batch_size: int = 128
    """batch size used when sampling from replay buffer"""
    max_buffer_capacity: int = 800000
    """maximum number of transitions stored; FIFO removes oldest episodes when exceeded"""
    episodic_replaybuffer: bool = False
    """flag for using an episodic replaybuffer for sequential storage of env steps"""

    # ===============================
    # Evaluation Parameters
    # ===============================

    evaluate_period: int = 2
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


    # =============================================================================================================================
    # Robust Ant Environment Parameters 
    # =============================================================================================================================

    # =========================
    # Base Ant reward settings
    # =========================
    forward_reward_weight: float = 1.0
    """Weight for forward progress reward."""
    healthy_reward_weight: float = 1.2
    """Weight for alive/healthy reward. Use e.g. 0.8 for multi-task setups."""
    ctrl_cost_weight: float = 0.5
    """Weight for control cost term."""
    contact_cost_weight: float = 5e-4
    """Weight for contact cost term."""

    # ==================================
    # Task mode / environment behaviour
    # ==================================
    task_mode: str = "default"
    """Options:
    - 'default': run forward
    - 'inverted_without_task_hint': meta-learning test, run only in two directions without task hint
    - 'target_goal': move to target XY position, with task hint
    """
    include_cfrc_ext_in_observation: bool = False
    """Whether to include external contact force terms in the observation."""
    history_len: int = 5
    """lenght of sequence of actions and rewards which are 
    stacked in obs for inverted_without_task_hint"""

    # ============================
    # Velocity task reward shaping
    # ============================
    velocity_reward_scale: float = 1.5
    """Scaling factor for velocity-matching reward."""
    scale_wrong_direction_reward: float = 1.5
    """Penalty/reduction scaling when moving in the wrong direction."""
    vel_rew_max_speed: float = 4.0
    """Maximum speed considered in the velocity reward."""
    append_task_reward: bool = False
    """whether to use the task specific reward as observation in inverted_without_task_hint or total reward"""

    # ==========================
    # Target / position task
    # ==========================
    position_reward_scale: float = 5.0
    """Scaling factor for target-position reward."""
    success_radius: float = 2.0
    """Distance threshold for task success."""
    maximum_area: float = 100.0
    """Maximum target sampling area / workspace bound."""

    # ==========================
    # General auxiliary rewards
    # ==========================
    movement_bonus_scale: float = 0.1
    """Bonus for movement in any direction."""
    tilt_penalty_weight: float = 0.5
    """Penalty weight for excess torso tilt beyond threshold."""
    death_penalty: float = -50.0
    """Penalty applied when the agent dies / becomes unhealthy."""


    # =========================================
    # Domain randomization / meta-learning setup
    # =========================================
    rand_mode: str = "default"
    """Environment parameter randomization mode.
    Currently supports:
    - 'RAO'
    - 'RFI'
    - None: no parameter randomization
    """
    rand_split_ratio: float = 0.5
    """Population split ratio used in ERFI mode.
    Example: 0.9 means first 90% use RFI and last 10% use RAO.
    """
    noise_limit: float = 0.1
    """Noise limit for RFI and RAO randomization."""

