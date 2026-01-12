
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

    def target_critic_forward_pass(
        self, 
        state_batch,            # [B, dim_obs]
        next_state_batch,       # [B, dim_obs] 
        all_sampled_actions,         # [sample_num, 2*B]
        ):

        # --- state shapes ---
            assert state_batch.ndim == 2, f"state_batch must be (B, obs_dim), got {tuple(state_batch.shape)}"
            assert next_state_batch.shape == state_batch.shape, \
                f"next_state_batch {tuple(next_state_batch.shape)} != state_batch {tuple(state_batch.shape)}"
            assert state_batch.shape[1] == self.state_dim, \
                f"state_dim mismatch: state_batch.shape[1]={state_batch.shape[1]} != {self.state_dim}"

            B = state_batch.size(0)

            # --- action shapes ---
            assert all_sampled_actions.ndim == 3, \
                f"all_sampled_actions must be (N, 2B, act_dim), got {tuple(all_sampled_actions.shape)}"

            N, twoB, A = all_sampled_actions.shape
            assert N == self.sample_action_num, \
                f"N mismatch: all_sampled_actions N={N} != self.sample_action_num={self.sample_action_num}"
            assert twoB == 2 * B, \
                f"2B mismatch: all_sampled_actions.shape[1]={twoB} != 2*B={2*B}"
            assert A == self.action_dim, \
                f"act_dim mismatch: all_sampled_actions.shape[2]={A} != self.action_dim={self.action_dim}"

            with torch.no_grad():
                all_states = torch.cat([state_batch, next_state_batch], dim=0)  # (2B, obs_dim)
                expanded_all_states = all_states.unsqueeze(0).expand(N, -1, -1)                  # (N, 2B, obs)  

                all_target_q = self.target_critic.forward(
                    expanded_all_states.reshape(-1, self.state_dim),    # (N* 2B, action_dim)
                    all_sampled_actions.reshape(-1, self.action_dim)    # (N * 2B, action_dim)
                ).reshape(N, 2*B)  # (sample_num, 2B)

                target_q = all_target_q[:,:B]
                next_target_q = all_target_q[:,B:]

            return target_q, next_target_q


    
    def expectation_step(self, target_q, sampled_actions, collect_stats: bool=False):
        """
        E-step of MPO:
        - Sample actions from the target policy for each state.
        - Evaluate Q-values with the target critic.
        - Update the temperature parameter eta via the dual function.
        - Compute normalized weights over actions (norm_target_q).
        """
        diff_out_of_bound = None
        
        if self.eta_dual.grad is not None:
            self.eta_dual.grad = None
        if self.use_action_penalty and self.eta_penalty.grad is not None:
            self.eta_penalty.grad = None

        # Dual function returns a scalar that we differentiate w.r.t. eta
        norm_target_q, loss_temperature = self.compute_weights_temperature_loss(self.eta_dual, target_q, self.eps_dual)
        loss_dual = loss_temperature

        has_oob = None
        penalty_normalized_weights = None
        loss_penalty_temperature = None
        diff_out_of_bound = None
        
        stats = None
        if collect_stats:
            stats = {
                "eta_dual": self.eta_dual.detach(),
                "norm_target_q_mean": norm_target_q.detach().mean(),
                "norm_target_q_min":  norm_target_q.detach().amin(),
                "norm_target_q_max":  norm_target_q.detach().amax(),
                "loss_dual":          loss_dual.detach(),
            }

        if self.use_action_penalty:
            # Compute action penalty when out of bound:
            actions_squashed  = torch.max(torch.min(sampled_actions, self.action_space_high), self.action_space_low)
            diff_out_of_bound = sampled_actions - actions_squashed
            has_oob = (diff_out_of_bound.abs().amax() > 0).to(norm_target_q.dtype)
            
            cost_out_of_bound = -(diff_out_of_bound.pow(2).sum(dim=-1))  # (N,B)
            penalty_normalized_weights, loss_penalty_temperature = self.compute_weights_temperature_loss(
                self.eta_penalty, cost_out_of_bound, self.eps_penalty
                )

            lam_eff = self.lam * has_oob

            norm_target_q = (1 - lam_eff) * norm_target_q + lam_eff * penalty_normalized_weights
            norm_target_q = norm_target_q / (norm_target_q.sum(dim=0, keepdim=True) + 1e-8)
            loss_dual = (1 - lam_eff) * loss_dual + lam_eff * loss_penalty_temperature
            

        loss_dual.backward()

        with torch.no_grad():
            # Gradient descent update on eta
            self.eta_dual.add_(-self.eta_lr * self.eta_dual.grad)
            self.eta_dual.clamp_(min=1e-3)
            self.eta_dual.grad = None
            
            if self.use_action_penalty:
                # if has_oob=0, grad is effective 0 cause lam_eff=0 -> no Update
                self.eta_penalty.add_(-self.eta_penalty_lr * self.eta_penalty.grad)
                self.eta_penalty.clamp_(min=1e-3)
                self.eta_penalty.grad = None

    

            if self.use_action_penalty and collect_stats:
                stats.update({
                    "eta_penalty": self.eta_penalty.detach(),
                    "has_out_of_bound": has_oob.detach() if has_oob is not None else None,
                    "diff_out_abs_mean": diff_out_of_bound.detach().abs().mean(),
                    "diff_out_abs_max":  diff_out_of_bound.detach().abs().amax(),
                    "penalty_weights_mean": penalty_normalized_weights.detach().mean(),
                    "penalty_weights_min":  penalty_normalized_weights.detach().amin(),
                    "penalty_weights_max":  penalty_normalized_weights.detach().amax(),
                    "norm_weights_mean":  norm_target_q.detach().mean(),
                    "norm_weights_min":  norm_target_q.detach().amin(),
                    "norm_weights_max":  norm_target_q.detach().amax(),
                    "loss_penalty":         loss_penalty_temperature.detach() if loss_penalty_temperature is not None else None,
                })

        return norm_target_q, stats
    

    def maximization_step(self, state_batch, norm_target_q, sampled_actions, b_mu, b_std, collect_stats): 
        """
        M-step of MPO:
        - Update the policy parameters to maximize the weighted log-likelihood
        under the non-parametric target distribution (defined by norm_target_q).
        - Enforce KL constraints between the new and old policy via Lagrange multipliers.
        """
        # Check for correct dimensions
        self.assert_sampled_actions_shape(sampled_actions, state_batch)

        # Current actor parameters for this batch of states
        mu, std = self.actor.forward(state_batch)      

        pi_1 = Independent(Normal(mu, b_std), 1)
        pi_2 = Independent(Normal(b_mu, std), 1)
        logp1 = pi_1.log_prob(sampled_actions)
        logp2 = pi_2.log_prob(sampled_actions)

        # Weighted policy improvement objective
        weighted_logp = (norm_target_q * (logp1 + logp2)) 
        self.loss_p = - weighted_logp.sum(dim=0).mean()        # sum over N -> (B,), mean over B -> scalar

        # KL constraints between old and new Gaussian policies
        C_mu_dim, C_sigma_dim, _, _ = gaussian_kl_diag(b_mu, mu, b_std, std)

        with torch.no_grad():

            # Update lagrange multipliers by gradient descent
            self.eta_mu -= self.alpha_mu_scale * (self.eps_mu_dim - C_mu_dim)
            self.eta_sigma -= self.alpha_sigma_scale * (self.eps_gamma_dim - C_sigma_dim)

            # Clamp Lagrange multipliers to a reasonable range
            self.eta_mu.clamp_(1e-10, self.alpha_mu_max)
            self.eta_sigma.clamp_(1e-10, self.alpha_sigma_max)

        # Total actor loss: maximize weighted log-prob subject to KL constraints
        constraint_mu = (C_mu_dim - self.eps_mu_dim)
        constraint_sigma = (C_sigma_dim - self.eps_gamma_dim)

        self.loss_l = self.loss_p \
              + (self.eta_mu * constraint_mu).sum() \
              + (self.eta_sigma * constraint_sigma).sum()

        self.actor_optimizer.zero_grad(set_to_none=True)
        self.loss_l.backward()
        clip_grad_norm_(self.actor.parameters(), 0.1)
        self.actor_optimizer.step() 

        if collect_stats:

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
        else:
            stats = None

        return stats

    def critic_update_td(self, next_target_q,  state_batch, action_batch, reward_batch, terminated_batch, truncated_batch, collect_stats):
        B = state_batch.size(0)
        reward_batch = reward_batch.view(-1)
        terminated_batch = terminated_batch.view(-1)
        truncated_batch  = truncated_batch.view(-1)

        # check for correct dimensions
        assert next_target_q.shape == (self.sample_action_num, B), \
            f"next_target_q shape {tuple(next_target_q.shape)} expected {(self.sample_action_num, B)}"
        assert reward_batch.shape == (B,), f"reward_batch shape {tuple(reward_batch.shape)} expected {(B,)}"
        assert terminated_batch.shape == (B,), f"terminated_batch shape {tuple(terminated_batch.shape)} expected {(B,)}"
        assert truncated_batch.shape == (B,), f"truncated_batch shape {tuple(truncated_batch.shape)} expected {(B,)}"
        assert state_batch.ndim == 2 and state_batch.shape[0] == B and state_batch.shape[1] == self.state_dim, \
            f"state_batch shape {tuple(state_batch.shape)} expected {(B, self.state_dim)}"
        assert action_batch.ndim == 2 and action_batch.shape[0] == B and action_batch.shape[1] == self.action_dim, \
            f"action_batch shape {tuple(action_batch.shape)} expected {(B, self.action_dim)}"

        with torch.no_grad():
            
            expected_next_q = next_target_q.mean(dim=0) #[B,]
            
            bootstrap_mask = 1.0 - terminated_batch  # (B,)
            q_target = reward_batch + bootstrap_mask * self.gamma * expected_next_q
            assert q_target.shape == (B,), f"q_target shape {tuple(q_target.shape)} expected {(B,)}"

        self.critic_optimizer.zero_grad(set_to_none=True)
        q_current = self.critic(state_batch, action_batch).squeeze()
        assert q_current.shape == (B,), f"q_current shape {tuple(q_current.shape)} expected {(B,)}"
        loss = self.norm_loss_q(q_target, q_current)
        loss.backward()
        self.critic_optimizer.step()

        if collect_stats:
            # Stats
            critic_loss = loss.detach()
            q_current = q_current.mean().detach()
            q_target = q_target.mean().detach()
           
            stats = {
                "critic_loss":  critic_loss,
                "q_current_mean":    q_current,
                "q_target_mean":     q_target,
                "terminated_rate": terminated_batch.mean().detach(),
                "truncated_rate": truncated_batch.mean().detach(),

            }
        else:
            stats= None

        return stats

    def sample_actions_from_target_actor(self, state_batch, next_state_batch = None, sample_num = 20):
        B = state_batch.size(0)
        with torch.no_grad():

            if next_state_batch is not None:
                all_states = torch.cat([state_batch, next_state_batch], dim=0)  # (2B, obs_dim)

                # get distribution
                all_sampled_actions, b_mu, b_std = self.target_actor.sample_action(all_states, sample_num) # (2B,)
                all_sampled_actions  = all_sampled_actions.permute( 1, 0, 2).contiguous()    #(sample_num, 2B, action_dim)
                sampled_actions      = all_sampled_actions[:, :B]  #(sample_num, B, action_dim)
                sampled_next_actions = all_sampled_actions[:, B:]  #(sample_num, B, action_dim)
                return  all_sampled_actions, sampled_actions, b_mu[:B], b_std[:B]
            else:
                sampled_actions, b_mu, b_std = self.target_actor.sample_action(state_batch, sample_num)
                sampled_actions = sampled_actions.permute(1,0,2)
                return sampled_actions, b_mu, b_std



    
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
    

    
    def compute_weights_temperature_loss(self, temperature: torch.Tensor, values: torch.Tensor, epsilon: float):
        """
        temperature: torch scalar, requires_grad=True
        values: (N, B)  (keine Gradienten; typischerweise aus target critic, detached)
        epsilon: float
        returns:
        weights: (N, B) detached
        dual_loss: scalar tensor (hat Grad wrt temperature)
        """
        values = values.detach()  # stop grad for 

        # weights (stop grad wrt temperature) 
        tempered = values / temperature.detach()
        tempered = tempered - tempered.max(dim=0, keepdim=True).values  # stable
        weights = torch.softmax(tempered, dim=0).detach()  # detach == stop-gradient

        # dual loss (Grad wrt temperature) 
        values_T = values.transpose(0, 1)  # (B, N)
        max_v = values_T.max(dim=1, keepdim=True).values  # (B, 1)
        exp_term = torch.exp((values_T - max_v) / temperature)  # grad fließt in temperature
        log_mean_exp = torch.log(exp_term.mean(dim=1) + 1e-8)  # (B,)

        dual_loss = temperature * epsilon + max_v.mean() + temperature * log_mean_exp.mean()


        return weights, dual_loss
    
    # def dual_function(self, eta, target_q):
    #     """
    #     eta: scalar Tensor, requires_grad=True
    #     target_q: Tensor shape (N, K)
    #     eps_dual: float (epsilon aus Paper)
    #     """
    #     tq = target_q.transpose(0, 1)          # (K, N)
    #     max_q, _ = tq.max(dim=1, keepdim=True)  # (K, 1)
    #     exp_term = torch.exp((tq - max_q) / eta)       # (K, N)
    #     inner_mean = exp_term.mean(dim=1)              # (K,)
    #     log_term = torch.log(inner_mean)               # (K,)
        
    #     dual_value = (
    #         eta * self.eps_dual
    #         + max_q.squeeze(1).mean()                  # mean der max_q über K
    #         + eta * log_term.mean()
    #     )
    #     return dual_value

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
