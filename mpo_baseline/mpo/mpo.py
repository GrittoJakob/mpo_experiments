
import os
from time import sleep
import numpy as np
from scipy.optimize import minimize
from tqdm import tqdm
import torch
import gymnasium as gym
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.distributions import MultivariateNormal, Independent, Normal
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from nets.actor import Actor
from nets.critic import Critic
from buffer.replaybuffer import ReplayBuffer
import time
import glob
import wandb
import math
from helpers.utils import gaussian_kl, gaussian_kl_diag

        
class MPO(object):
    def __init__(
            self, 
            args, 
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

        # Optimizer
        self.actor_optimizer = actor_optimizer
        self.critic_optimizer =  critic_optimizer

        # Environment & basic dimensions
        self.state_dim = args.obs_space
        self.action_dim = args.action_dim

        # Device used for tensors / networks (CPU or GPU)
        self.device = device

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

        self.log_dir = args.log_dir
        self.model_dir = os.path.join(self.log_dir, "model")
        os.makedirs(self.model_dir, exist_ok=True)

        # Choose Q-loss type (MSE or Smooth L1)
        self.norm_loss_q = nn.MSELoss() if args.q_loss_type == 'mse' else nn.SmoothL1Loss()

                          
        # Dual variables / Lagrange multipliers and counters
        # Temperature parameter for MPO E-step (initialized randomly)
        self.eta = np.random.rand()

        # Lagrange multipliers for KL constraints in the M-step
        self.eta_mu = torch.full((self.action_dim,), args.init_eta_mu, device=self.device,dtype=torch.float32)
        self.eta_sigma = torch.full((self.action_dim,), args.init_eta_sigma, device=self.device, dtype=torch.float32)

    
    def expectation_step(self, state_batch, sampled_action = None, b_mu = None, b_A= None):
        """
        E-step of MPO:
        - Sample actions from the target policy for each state.
        - Evaluate Q-values with the target critic.
        - Update the temperature parameter eta via the dual function.
        - Compute normalized weights over actions (norm_target_q).
        """

        B = state_batch.shape[0]  # the sample number of states
        sample_num = self.sample_action_num  # the sample number of actions per state
        
        with torch.no_grad():

            if sampled_action is None:
                
                
                # Get mean and Cholesky factor of the target policy for each state
                b_mu, b_A = self.target_actor.forward(state_batch)  # (B,)

                # Gaussian policy over actions with batch size K
                b = Independent(Normal(b_mu, b_A), 1)  # (B,)

                # Sample N actions per state -> shape (sample_num, B, action_dim)
                sampled_actions = b.sample((sample_num,))  
                
            else:
                sampled_actions = sampled_action

                #Check for correct dimensions
                self.assert_sampled_actions_shape(sampled_actions, state_batch)
            
            expanded_states = state_batch[None, ...].expand(sample_num, -1, -1)  # (sample_num, B, state_dim)
            
            # Evaluate Q-values of the target critic for all (state, action) pairs
            target_q = self.target_critic.forward(
                expanded_states.reshape(-1, self.state_dim), sampled_actions.reshape(-1, self.action_dim)  # (N * K, action_dim)
            ).reshape(sample_num, B)
        
        # Treat target_q as constant w.r.t. eta (no gradients from critic/actor)
        eta_tensor = torch.tensor(self.eta, device=self.device, requires_grad=True)

        # Dual function returns a scalar that we differentiate w.r.t. eta
        dual_val = self.dual_function(eta_tensor, target_q)
        dual_val.backward()
        with torch.no_grad():
            # Gradient descent update on eta
            eta_tensor -= self.eta_lr * eta_tensor.grad
            eta_tensor.clamp_(min=1e-6)
            self.eta = eta_tensor.item()
        
        # Compute normalized weights over actions for each state
        # Shape stays (sample_num, B), softmax along action-sample dimension sample_num
        norm_target_q = torch.softmax(target_q / self.eta, dim=0)  # (sample_num, B) or (action_dim, B)

        return sampled_actions, norm_target_q, b_mu, b_A, self.eta
    

    def maximization_step(self, state_batch, norm_target_q, sampled_actions, b_mu, b_A): 
        """
        M-step of MPO:
        - Update the policy parameters to maximize the weighted log-likelihood
        under the non-parametric target distribution (defined by norm_target_q).
        - Enforce KL constraints between the new and old policy via Lagrange multipliers.
        """
        # Check for correct dimensions
        self.assert_sampled_actions_shape(sampled_actions, state_batch)

        # Current actor parameters for this batch of states
        mu, A = self.actor.forward(state_batch)

        # paper1 version
        #policy = MultivariateNormal(loc=mu, scale_tril=A)  # (B,)
        #loss_p = torch.mean( norm_target_q * policy.expand((N, K)).log_prob(sampled_actions))  # (N, B)
        #C_mu, C_sigma, sigma_i_det, sigma_det = gaussian_kl( mu_i=b_mu, mu=mu, Ai=b_A, A=A)

        # Two Gaussian policies as in the "paper2" variant:
        # pi_1 uses new mean and old covariance; pi_2 uses old mean and new covariance.

        # b_std = torch.diagonal(b_A, dim1=-2, dim2=-1)  # (B, da)
        # std   = torch.diagonal(A,  dim1=-2, dim2=-1)  # (B, da)
        b_std = b_A
        std = A

        pi_1 = Independent(Normal(mu, b_std), 1)
        pi_2 = Independent(Normal(b_mu, std), 1)
        logp1 = pi_1.log_prob(sampled_actions)
        logp2 = pi_2.log_prob(sampled_actions)

        # Weighted policy improvement objective
        self.loss_p = torch.mean(norm_target_q * (logp1 + logp2))

        # KL constraints between old and new Gaussian policies
        C_mu_dim, C_sigma_dim, _, _ = gaussian_kl_diag(b_mu, mu, b_A, A)

        with torch.no_grad():

            # Update lagrange multipliers by gradient descent
            self.eta_mu -= self.alpha_mu_scale * (self.eps_mu_dim - C_mu_dim)
            self.eta_sigma -= self.alpha_sigma_scale * (self.eps_gamma_dim - C_sigma_dim)

            # Clamp Lagrange multipliers to a reasonable range
            self.eta_mu.clamp_(1e-10, self.alpha_mu_max)
            self.eta_sigma.clamp_(1e-10, self.alpha_sigma_max)

        # Total actor loss: maximize weighted log-prob subject to KL constraints
        self.actor_optimizer.zero_grad()
        self.loss_l = -( self.loss_p + (self.eta_mu * (self.eps_mu_dim - C_mu_dim)).sum() + (self.eta_sigma * (self.eps_gamma_dim - C_sigma_dim)).sum())
        self.loss_l.backward()
        clip_grad_norm_(self.actor.parameters(), 0.1)
        self.actor_optimizer.step() 

        # LOGGING STATS 
        C_mu_mean    = C_mu_dim.mean().detach()
        C_sigma_mean = C_sigma_dim.mean().detach()

        eta_mu_mean    = self.eta_mu.mean().detach()
        eta_mu_min     = self.eta_mu.min().detach()
        eta_mu_max     = self.eta_mu.max().detach()
        eta_sigma_mean = self.eta_sigma.mean().detach()
        eta_sigma_min  = self.eta_sigma.min().detach()
        eta_sigma_max  = self.eta_sigma.max().detach()
        std_mean       = std.mean().detach()
        mu_mean        = mu.mean().detach()
        std_dim_mean = std.mean(dim=(0, 1))   # -> [A]
        mu_dim_mean  = mu.mean(dim=(0, 1))    # -> [A]  
        std_min = std_dim_mean.min().detach()
        std_max = std_dim_mean.max().detach()

        mu_min = mu_dim_mean.min().detach()
        mu_max = mu_dim_mean.max().detach()

        stats = {
            "loss_p":        self.loss_p.detach(),
            "loss_l":        self.loss_l.detach(),
            "C_mu_mean":     C_mu_mean,
            "C_sigma_mean":  C_sigma_mean,
            "eta_mu_mean":   eta_mu_mean,
            "eta_mu_min":    eta_mu_min,
            "eta_mu_max":    eta_mu_max,
            "eta_sigma_mean":eta_sigma_mean,
            "eta_sigma_min": eta_sigma_min,
            "eta_sigma_max": eta_sigma_max,
            "std_mean":      std_mean,
            "mu_mean":       mu_mean,
            "mu_max":        mu_max,
            "mu_min":        mu_min,
            "std_max":       std_max,
            "std_min":       std_min,
        }

        return stats

    def critic_update_td(self, state_batch, action_batch, next_state_batch, reward_batch, sample_num=32):
        B = state_batch.size(0)
    
        with torch.no_grad():

            ## get mean, cholesky from target actor --> to sample from Gaussian
            
            #!concatenation of all steps to perform only one forward pass of action sampling to reduce computional costs!
            all_states = torch.cat([state_batch, next_state_batch], dim=0)  # (2B, obs_dim)

            pi_mean, pi_std = self.target_actor.forward(all_states)  # (2B,)
            policy = Independent(Normal(pi_mean, pi_std), 1)  # (2B,)

            all_sampled_actions = policy.sample((sample_num,)).permute(1, 0, 2)  # (2B, sample_num, action_dim)
            sampled_actions      = all_sampled_actions[:B]  # (B, sample_num, action_dim)
            sampled_next_actions = all_sampled_actions[B:]  # (B, sample_num, action_dim)
            b_mu = pi_mean[:B]
            b_A =pi_std[:B]

            # Permute sampled_actions for E-Step:
            sampled_actions_NBA =  sampled_actions.permute(1, 0,2)    # ( sample_num, B, action_dim)

            expanded_next_states = next_state_batch[:, None, :].expand(-1, sample_num, -1)  # (2B, sample_num, state_dim)
            
            ## get expected Q value from target critic
            expected_next_q = self.target_critic.forward(
                expanded_next_states.reshape(-1, self.state_dim),  # (B * sample_num, state_dim)
                sampled_next_actions.reshape(-1, self.action_dim)  # (B * sample_num, action_dim)
            ).reshape(B, sample_num).mean(dim=1)  # (B,)
            
            y = reward_batch + self.gamma * expected_next_q
        self.critic_optimizer.zero_grad()
        t = self.critic(state_batch, action_batch).squeeze()
        loss = self.norm_loss_q(y, t)
        loss.backward()
        self.critic_optimizer.step()

        # Stats
        critic_loss = loss.detach()
        q_current = t.mean().detach()
        q_target = y.mean().detach()

        stats = {
            "critic_loss":  critic_loss,
            "q_current_mean":    q_current,
            "q_target_mean":     q_target
        }

        return stats, sampled_actions_NBA, b_mu, b_A


    # ## Mistake: need to change buffer for storage of old policies!
    # def critic_update_retrace(self, state_batch, action_batch, next_state_batch, reward_batch, sample_num=64):
       
    #     with torch.no_grad():

    #         log_prob = self.actor.evaluate_action(state_batch, action_batch)
    #         target_log_prob = self.target_actor.evaluate_action(state_batch, action_batch)

    #         c_ret = self.calc_retrace_weights(target_log_prob, log_prob)

    #         with torch.no_grad():
    #             print("🔍 c_ret stats — min:", c_ret.min().item(), 
    #                 "max:", c_ret.max().item(), 
    #                 "mean:", c_ret.mean().item())

    #         Q_target = self.target_critic(state_batch, action_batch).squeeze(-1)
    #           # [B, N, act_dim]
    #         sampled_actions = self.target_actor.sample_action(next_state_batch, sample_num=sample_num)  # [B, N, act_dim]
    #         s_next_expanded = next_state_batch.unsqueeze(1).expand(-1, sample_num, -1)  # [B, N, obs_dim]
            
    #         flat_s = s_next_expanded.reshape(-1, next_state_batch.shape[-1])
    #         flat_a = sampled_actions.reshape(-1, action_batch.shape[-1])
    #         flat_q = self.target_critic.forward(flat_s, flat_a).squeeze(-1)  # [B*N]

    #         q_values = flat_q.view(state_batch.shape[0], sample_num)  # [B, N]
    #         expected_target_Q = q_values.mean(dim=1)  # [B]
            
    #         Q_ret = reward_batch + self.gamma * (expected_target_Q + c_ret * (expected_target_Q - Q_target))

    #     current_Q = self.critic(state_batch, action_batch).squeeze(-1)  # [B]
        
    #     critic_loss = self.norm_loss_q(current_Q, Q_ret)
    #     self.critic_optimizer.zero_grad()
    #     critic_loss.backward()
    #     self.critic_optimizer.step()

    #     return critic_loss, current_Q, Q_ret

    
    def assert_sampled_actions_shape(self, sampled_actions, state_batch, name="sampled_actions"):
        """
        Ensures sampled_actions has shape (N, B, action_dim)
        where:
        sample_num = number of sampled actions per state
        B = state_batch.size(0)
        action_dim = self.action_dim
        """

        assert sampled_actions is not None, f"{name} is None"
        assert sampled_actions.dim() == 3, \
            f"{name} must be 3D (sample_num, B, act_dim), got shape {tuple(sampled_actions.shape)}"

        N, B, A = sampled_actions.shape
        expected_B = state_batch.size(0)

        assert N == self.sample_action_num, \
            f"{name}: Expected N={self.sample_action_num} actions per state, got N={N}"
        assert B == expected_B, \
            f"{name}: Expected B={expected_B} states, got K={B}"
        assert A == self.action_dim, \
            f"{name}: Expected action_dim={self.action_dim}, got A={A}"



    def update_target_actor_critic(self):
        # param(target_actor) <-- param(actor)
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)

        # param(target_critic) <-- param(critic)
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)
    
    def dual_function(self, eta, target_q):
        """
        eta: scalar Tensor, requires_grad=True
        target_q: Tensor shape (N, K)
        eps_dual: float (epsilon aus Paper)
        """
        tq = target_q.transpose(0, 1)          # (K, N)
        max_q, _ = tq.max(dim=1, keepdim=True)  # (K, 1)
        exp_term = torch.exp((tq - max_q) / eta)       # (K, N)
        inner_mean = exp_term.mean(dim=1)              # (K,)
        log_term = torch.log(inner_mean)               # (K,)
        
        dual_value = (
            eta * self.eps_dual
            + max_q.squeeze(1).mean()                  # mean der max_q über K
            + eta * log_term.mean()
        )
        return dual_value


    # def load_model(self, path=None):
    #     load_path = path if path is not None else self.save_path
    #     with torch.serialization.safe_globals([np.core.multiarray.scalar]):
    #         checkpoint = torch.load(load_path, weights_only=False)
  
       
    #     self.critic.load_state_dict(checkpoint['critic'])
    #     self.target_critic.load_state_dict(checkpoint['target_critic'])
    #     self.actor.load_state_dict(checkpoint['actor'])
    #     self.target_actor.load_state_dict(checkpoint['target_actor'])
    #     self.critic_optimizer.load_state_dict(checkpoint['critic_optim'])
    #     self.actor_optimizer.load_state_dict(checkpoint['actor_optim'])

    #     # load Lagrange multipliers
    #     self.eta = checkpoint["eta"]
    #     self.eta_mu = checkpoint["eta_mu"]
    #     self.eta_sigma = checkpoint["eta_sigma"]
                
    #     # replay buffer
    #     if "replay_buffer" in checkpoint:
    #         self.replaybuffer.load_state_dict(checkpoint["replay_buffer"])

    #     self.critic.train()
    #     self.target_critic.eval()
    #     self.actor.train()
    #     self.target_actor.eval()

    # def save_lightweight(self, path, iteration):
    #     """Fast checkpoint: networks + optimizers only."""
    #     data = {
    #         "iteration": iteration,
    #         "actor": self.actor.state_dict(),
    #         "critic": self.critic.state_dict(),
    #         "target_actor": self.target_actor.state_dict(),
    #         "target_critic": self.target_critic.state_dict(),
    #         "actor_optim": self.actor_optimizer.state_dict(),
    #         "critic_optim": self.critic_optimizer.state_dict(),

    #         # Lagrange multipliers
    #         "eta": self.eta,
    #         "eta_mu": self.eta_mu,
    #         "eta_sigma": self.eta_sigma
    #     }
    #     torch.save(data, path)

    # def save_full(self, path,iteration):
    #     """Full checkpoint including replay buffer."""
    #     data = {
    #         "iteration": iteration,
    #         "actor": self.actor.state_dict(),
    #         "critic": self.critic.state_dict(),
    #         "target_actor": self.target_actor.state_dict(),
    #         "target_critic": self.target_critic.state_dict(),
    #         "actor_optim": self.actor_optimizer.state_dict(),
    #         "critic_optim": self.critic_optimizer.state_dict(),

    #         # Lagrange multipliers
    #         "eta": self.eta,
    #         "eta_mu": self.eta_mu,
    #         "eta_sigma": self.eta_sigma,

    #         # Replay buffer
    #         "replay_buffer": self.replaybuffer.state_dict()
    #     }
    #     tmp = path + ".tmp"
    #     torch.save(data, tmp)
    #     os.replace(tmp, path)

    # def save(self, iteration):
    #     """Main save function controlled by configuration."""
    #     # 1) ALWAYS save a lightweight latest model
    #     if self.save_latest:
    #         self.save_lightweight(
    #             os.path.join(self.model_dir, "latest_model_without_rb.pt"),
    #             iteration
    #         )

    #     # 2) EVERY N iterations save a full snapshot
    #     if iteration % self.save_every == 0:
            
    #         if self.save_replay_buffer:
    #             path = os.path.join(self.model_dir, "full_backup.pt")
    #             if os.path.exists(path):
    #                 os.remove(path)
    #             self.save_full(path,iteration)
    #         else:
    #             self.save_lightweight(os.path.join(self.model_dir, "latest_model_without_rb.pt"))
