from dataclasses import dataclass
from .base_args import BaseArgs

@dataclass
class Ant_Maze_Args(BaseArgs):

    # Environment

    env_id: str = "AntMaze_UMaze-v5"
    """gym environment name (used in gym.make)"""
    seed: int = 1
    """seed of the experiment"""
    exclude_contact_forces: bool = True
    """exlude contact forces"""

    # Actor & Critic
    hidden_size_actor: int = 128
    """hidden size of actor network"""
    hidden_size_critic: int = 128
    """hidden size of critc network"""
    actor_lr: float = 3e-4
    """Learning rate for actor Adam optimizer"""
    critic_lr: float = 3e-4
    """Learning rate for critic adam optimizer"""   

    # Train Loop
    UTD_ratio: float = 0.5
    """ Ratio: num_updates per env step"""
    max_training_steps: int = 500000
    """Maximal number of env steps for training"""

    # Sampling / Replay Buffer

    batch_size: int = 256
    """batch size used when sampling from replay buffer"""
    max_buffer_capacity: int = 100000
    """maximum number of transitions stored; FIFO removes oldest episodes when exceeded"""
    episodic_replaybuffer: bool = False
    """flag for using an episodic replaybuffer for sequential storage of env steps"""