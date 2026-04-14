import os
import types
import torch
import torch.nn as nn
from nets.MLP_actor import Actor
from nets.MLP_critic import Critic

from .expectation_step import expectation_step, compute_weights_temperature_loss
from .maximization_step import maximization_step
from .transitive_rl import trl_critic_loss
from environment.gc_wrappers.gc_obs_wrapper import GC_Obs_Wrapper


class TRL_with_MPO(object):
    
    def __init__(
            self, 
            args, 
            env,
            obs_wrapper,
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
        self.state_dim = args.obs_dim
        self.action_dim = args.action_dim

        # Device used for tensors / networks (CPU or GPU)
        self.device = device

        self.use_mass_force_KL = args.use_mass_force_KL

        # Number of action samples
        self.sample_action_num = args.sample_action_num
        self.batch_size = args.batch_size

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

        # TRL critic loss parametes
        self.expectile = args.expectile
        self.lamda_trl = args.lambda_trl

        # Wrapper
        self.obs_wrapper = obs_wrapper

        # Bindings of functions
        self.expectation_step = types.MethodType(expectation_step, self)
        self.maximization_step = types.MethodType(maximization_step, self)
        self.compute_weights_temperature_loss = types.MethodType(compute_weights_temperature_loss, self)
        self.trl_critic_loss = types.MethodType(trl_critic_loss, self)

    def update_target_actor_critic(self):
        # param(target_actor) <-- param(actor)
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)

        # param(target_critic) <-- param(critic)
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)

    def sample_actions_from_target_actor(self, state_batch, sample_num = 20):
        
        # This function samples actions from the target actor.
        # Inputs:
        #     - state-batch [B, obs_dim]
        
        # Returns:
        #     - sampled actions 
        #     - Mu and std from target actor
        
        with torch.no_grad():

            # get distribution
            sampled_actions, mu_off, std_off = self.target_actor.sample_action(state_batch, sample_num)
            sampled_actions = sampled_actions.permute(1,0,2) 
            mu_off  = mu_off.clone()
            std_off = std_off.clone()   
        
        return sampled_actions, mu_off, std_off
        

    def target_critic_forward_pass(self, obs_batch, sampled_actions):
        
        # Check correct dimensions
        N, B, A = sampled_actions.shape
        assert N == self.sample_action_num
        assert B == self.batch_size
        assert A == self.action_dim

        Bo, O = obs_batch.shape
        assert B == Bo
        assert O == self.state_dim

        expanded_states = obs_batch.unsqueeze(0).expand(N, -1,-1)   # (N, B, obs_dim)

        # Forward Pass
        target_q = self.target_critic.forward(
            expanded_states.reshape(-1, self.state_dim),
            sampled_actions.reshape(-1, self.action_dim)
        ).reshape(N, B)

        return target_q


