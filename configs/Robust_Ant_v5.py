from dataclasses import dataclass
from .base_args import BaseArgs

@dataclass
class Robust_Ant_Args(BaseArgs):

    # Environment
    env_id: str = "Ant-v5"
    """gym environment name (used in gym.make)"""
    seed: int = 1
    """seed of the experiment"""

    # Actor & Critic
    hidden_size_actor: int = 256
    """hidden size of actor network"""
    hidden_size_critic: int = 256
    """hidden size of critc network"""
    actor_lr: float = 3e-4
    """Learning rate for actor Adam optimizer"""
    critic_lr: float = 3e-4
    """Learning rate for critic adam optimizer"""   

    # Train Loop
    UTD_ratio: float = 0.5
    """ Ratio: num_updates per env step"""
    max_training_steps: int = 2000000
    """Maximal number of env steps for training"""

    # Sampling / Replay Buffer
    batch_size: int = 512
    """batch size used when sampling from replay buffer"""
    max_buffer_capacity: int = 800000
    """maximum number of transitions stored; FIFO removes oldest episodes when exceeded"""
    episodic_replaybuffer: bool = False
    """flag for using an episodic replaybuffer for sequential storage of env steps"""

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

    # ============================
    # Velocity task reward shaping
    # ============================
    velocity_reward_scale: float = 1.5
    """Scaling factor for velocity-matching reward."""
    scale_wrong_direction_reward: float = 1.5
    """Penalty/reduction scaling when moving in the wrong direction."""
    vel_rew_max_speed: float = 4.0
    """Maximum speed considered in the velocity reward."""

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


    # =========================================
    # E-Step Evaluation Distribution Parameters
    # ========================================= 
    eval_e_step: int = 3
    """how often to evaluate E-step distribution"""
    use_e_step_eval: bool = False
    """Flag for using new MPO_Learner for E-step evaluation"""
    sample_action_num_for_dist_eval: int = 32
    """how many samples for e-step evaluation"""
    m_step_eval_iterations: int = 4
    """number of iterations in the evaluation of the M-Step to show the fitting of the online actor to the auxiliar distribution"""