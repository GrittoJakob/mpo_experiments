import os
import types
import torch
import torch.nn as nn
from nets.actor import Actor
from nets.critic import Critic

from .critic_forward_pass import shared_target_critic_forward_pass
from .critic_update import critic_update_td
from .expectation_step import expectation_step, compute_weights_temperature_loss
from .maximization_step import maximization_step
from .sample_actions import sample_actions_from_target_actor


class MPO(object):
    def __init__(
            self, 
            args, 
            env,
            actor: Actor, 
            target_actor: Actor, 
            critic: Critic, 
            target_critic: Critic,
            actor_optimizer, 
            critic_optimizer,
            device
            ):
        """
        Main class implementing the MPO agent.
        Holds:
        - environment reference
        - actor / critic networks and their target networks
        """

        # Networks
        self.actor = actor
        self.target_actor = target_actor
        self.critic = critic
        self.target_critic = target_critic
        
        # Environment
        true_action_space = env.unwrapped.action_space
        self.action_space_low  = torch.as_tensor(true_action_space.low,  dtype=torch.float32, device=device)
        self.action_space_high = torch.as_tensor(true_action_space.high, dtype=torch.float32, device=device)

        print("Action Space: Low: ", self.action_space_low)
        print("Action Space: High: ", self.action_space_high)

        # Optimizer
        self.actor_optimizer = actor_optimizer
        self.critic_optimizer =  critic_optimizer

        # Environment & basic dimensions
        self.state_dim = args.obs_space
        self.action_dim = args.action_dim

        # Device used for tensors / networks (CPU or GPU)
        self.device = device

        self.use_mass_force_KL = args.use_mass_force_KL

        # Number of action samples
        self.sample_action_num = args.sample_action_num

        # MPO / optimization hyperparameters
        # Dual / KL constraints
        self.eps_dual = args.dual_constraint     # KL constraint for E-Step
        self.eps_mu_dim = torch.full((self.action_dim,), args.kl_mean_constraint, device = self.device, dtype=torch.float32)     # KL constraint for mean
        self.eps_gamma_dim = torch.full((self.action_dim,), args.kl_var_constraint, device = self.device, dtype=torch.float32)   # KL constraint for variance

        # Discount factor for returns
        self.gamma = args.discount_factor

        # Lagrange multiplier step sizes (for KL constraints)
        self.alpha_mu_scale = args.alpha_mean_scale     # scale for mean Lagrange multiplier update
        self.alpha_sigma_scale = args.alpha_var_scale   # scale for variance Lagrange multiplier update

        # Maximum values for Lagrange multipliers
        self.alpha_mu_max = args.alpha_mean_max
        self.alpha_sigma_max = args.alpha_var_max

        # Temperature learning rate (for dual variable eta)
        self.eta_lr = args.eta_lr

        # Dual variables / Lagrange multipliers and counters
        # Temperature parameter for MPO E-step (initialized randomly)
        self.eta_dual = torch.tensor(args.init_eta_dual, device=self.device, dtype=torch.float32, requires_grad=True)
        self.eta_penalty = torch.tensor(args.init_eta_dual, device=self.device, dtype=torch.float32, requires_grad=True)

        self.log_dir = args.log_dir
        self.model_dir = os.path.join(self.log_dir, "model")
        os.makedirs(self.model_dir, exist_ok=True)

        # Choose Q-loss type (MSE or Smooth L1)
        self.norm_loss_q = nn.MSELoss() if args.q_loss_type == 'mse' else nn.SmoothL1Loss()

        # Action penalty 
        self.use_action_penalty = args.use_action_penalty
        self.eps_penalty = getattr(args, "eps_penalty", 1e-3)     # ähnlich Acme default
        self.eta_penalty_lr = getattr(args, "eta_penalty_lr", self.eta_lr)
        self.lam = args.penalty_mix
                          
        # Lagrange multipliers for KL constraints in the M-step
        self.eta_mu = torch.full((self.action_dim,), args.init_eta_mu, device=self.device,dtype=torch.float32)
        self.eta_sigma = torch.full((self.action_dim,), args.init_eta_sigma, device=self.device, dtype=torch.float32)

        # Bindings of functions
        self.expectation_step = types.MethodType(expectation_step, self)
        self.maximization_step = types.MethodType(maximization_step, self)
        self.critic_update_td = types.MethodType(critic_update_td, self)
        self.sample_actions_from_target_actor = types.MethodType(sample_actions_from_target_actor, self)
        self.compute_weights_temperature_loss = types.MethodType(compute_weights_temperature_loss, self)
        self.shared_target_critic_forward_pass = types.MethodType(shared_target_critic_forward_pass, self)


    
    def update_target_actor_critic(self):
        # param(target_actor) <-- param(actor)
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)

        # param(target_critic) <-- param(critic)
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)
    