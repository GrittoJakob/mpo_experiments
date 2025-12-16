
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
from actor import Actor
from critic import Critic
from replaybuffer import ReplayBuffer
import time
import glob
import wandb
import math
from utils import gaussian_kl, gaussian_kl_diag
        
class MPO(object):
    def __init__(self, env, args, make_video_env):
        """
        Main class implementing the MPO agent.
        Holds:
        - environment reference
        - actor / critic networks and their target networks
        - replay buffer
        - all hyperparameters and logging buffers
        """
        # 1) Environment & basic dimensions
        self.env = env
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]

        # Device used for tensors / networks (CPU or GPU)
        self.device = args.device

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

        # Main learning rate for actor/critic optimizers
        self.learning_rate = args.learning_rate

        # How often we update eta (dual function)
        #self.update_dual_function_interval = args.update_dual_function_interval

        # 3) Sampling / training schedule

        self.max_training_steps = args.max_training_steps
        self.sample_steps_per_iter = args.sample_steps_per_iter     # Number of steps to sample per iteration       
        self.sample_episode_maxstep = args.sample_episode_maxstep   # Maximum steps per sampled episode
        self.sample_action_num = args.sample_action_num  # Number of action samples per state in MPO E-step
        self.batch_size = args.batch_size   # Minibatch size for training
        self.UTD_ratio = args.UTD_ratio # Update-to-data ratio (how many updates per collected step)
        self.delay_policy_update = args.delay_policy_update

        # Number of M-step (actor) iterations per E-step
        #self.mstep_iteration_num = args.mstep_iteration_num

        # Number of steps to warm up the replay buffer before training
        self.warm_up_steps = args.warm_up_steps

        # Evaluation / logging / saving schedule
        
        self.evaluate_period = args.evaluate_period     # How often to run evaluation (in training iterations)
        self.evaluate_episode_num = args.evaluate_episode_num    # Number of evaluation episodes
        self.evaluate_episode_maxstep = args.evaluate_episode_maxstep   # Max steps per evaluation episode

        # Logging behavior (e.g. WandB)
        self.wandb_track = args.track
        # Inner-loop logging interval (every N gradient updates)
        self.log_inner_interval = args.log_inner_interval

        # Model saving configuration
        self.save_every = args.save_every
        self.save_latest = args.save_latest
        self.log_dir = args.log_dir
        self.model_dir = os.path.join(self.log_dir, "model")
        os.makedirs(self.model_dir, exist_ok=True)

        # Policy / value networks and their targets
        self.std_init = args.std_init             # initial std for Gaussian policy
        self.use_retrace = args.use_retrace       # whether to use Retrace for critic update
        self.covariance_type = args.covariance_type

        # Main actor / critic networks
        self.actor = Actor(env, args.hidden_size_actor, self.std_init, self.covariance_type).to(self.device)
        self.critic = Critic(env, args.hidden_size_critic).to(self.device)

        # Target networks (used for stable targets)
        self.target_actor = Actor(env, args.hidden_size_actor, self.std_init, self.covariance_type).to(self.device)
        self.target_critic = Critic(env, args.hidden_size_critic).to(self.device)

        # Optional compile (PyTorch 2.x)
        if getattr(args, "use_compile", False):
            self.actor = torch.compile(self.actor)
            self.critic = torch.compile(self.critic)
            self.target_actor = torch.compile(self.target_actor)
            self.target_critic = torch.compile(self.target_critic)


        # How often to sync target networks with the main networks (in gradient updates)
        self.target_update_period = args.target_update_period

        # Initialize target networks with the same parameters as main networks
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)
            target_param.requires_grad = False     # target net is not trained directly

        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)
            target_param.requires_grad = False     # target net is not trained directly

        # Optimizers & loss functions
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), self.learning_rate)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), self.learning_rate)

        # Choose Q-loss type (MSE or Smooth L1)
        self.norm_loss_q = nn.MSELoss() if args.q_loss_type == 'mse' else nn.SmoothL1Loss()

        # Replay buffer
        self.replaybuffer = ReplayBuffer(args.max_replay_buffer)
        self.save_replay_buffer = args.save_replay_buffer  # whether to save buffer to disk (if implemented)
        self.q_update_step = 0      # counts critic updates

        self.buffer_size = 0                           
        # Dual variables / Lagrange multipliers and counters
        # Temperature parameter for MPO E-step (initialized randomly)
        self.eta = np.random.rand()

        # Lagrange multipliers for KL constraints in the M-step
        self.eta_mu = torch.full((self.action_dim,), args.init_eta_mu, device=self.device,dtype=torch.float32)
        self.eta_sigma = torch.full((self.action_dim,), args.init_eta_sigma, device=self.device, dtype=torch.float32)

        # Global step / iteration bookkeeping
        self.num_steps = 0
        self.start_iteration = 1
        self.global_update = 1
        
        # Vidoes 
        self.run_name = getattr(args, "run_name", "")
        self.render = args.render
        self.make_video_env = make_video_env
        self.video_dir = getattr(args, "video_dir", "videos")
        self.log_videos_period = args.log_videos_period
        self._video_cleaned_once = False

        # 9) Logging buffers (stats accumulated between log calls)
        # Critic-related statistics
        self.mean_loss_q = []
        self.mean_current_q = []
        self.mean_target_q = []

        # Actor-related statistics
        self.mean_loss_p = []
        self.mean_loss_l = []

        # KL and covariance statistics
        self.max_kl_mu = []
        self.max_kl_sigma = []
        self.min_kl_sigma  =[]
        self.min_kl_mu = []

        # Variance statistics
        self.var_mean = []
        self.var_min = []
        self.var_max = []

        # Runtime measurements (seconds)
        self.runtime_env = 0.0          # environment interaction time
        self.runtime_eval = 0.0         # evaluation time
        self.runtime_policy_eval = 0.0  # critic update time
        self.runtime_E_step = 0.0       # MPO E-step time
        self.runtime_M_step = 0.0       # MPO M-step time


    def sample_trajectory(self):
        """
        Collect a number of episodes by interacting with the environment
        using the current policy and store them in the replay buffer.
        Each episode is stored as a list of (state, action, next_state, reward) tuples.
        Returns:
        total_steps_collected (int): number of env steps actually executed
        """
        episodes = []
        total_steps_collected = 0
        self.actor.eval()
        
        with torch.no_grad():
            # Loop over episodes to collect
            while total_steps_collected < self.sample_steps_per_iter:
                buff = []
                state, info = self.env.reset()

                # Roll out one episode up to a maximum number of steps
                for _ in range(self.sample_episode_maxstep):
                    state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device)
                    
                    # Get action from actor
                    # NOTE: Actor.action already returns a NumPy array
                    action = self.actor.action(state_tensor)

                    # Step the environment
                    next_state, reward, terminated, truncated, info = self.env.step(action)
                    total_steps_collected += 1
                    
                    # Store transition in the current episode buffer
                    buff.append((state, action, next_state, reward))

                    if terminated or truncated:
                        break
                    state = next_state

                # Store completed episode
                episodes.append(buff)

        self.actor.train()

        # Push all collected episodes into the replay buffer and returns the number of collected steps
        self.replaybuffer.store_episodes(episodes)
        return total_steps_collected

    def expectation_step(self, state_batch, sampled_action = None, b_mu = None, b_A= None):
        """
        E-step of MPO:
        - Sample actions from the target policy for each state.
        - Evaluate Q-values with the target critic.
        - Update the temperature parameter eta via the dual function.
        - Compute normalized weights over actions (norm_target_q).
        """

        B = self.batch_size  # the sample number of states
        sample_num = self.sample_action_num  # the sample number of actions per state

        with torch.no_grad():

            if sampled_action is None:
                
                # Get mean and Cholesky factor of the target policy for each state
                b_mu, b_A = self.target_actor.forward(state_batch)  # (B,)

                # Gaussian policy over actions with batch size K
                b = MultivariateNormal(b_mu, scale_tril=b_A)  # (B,)

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

        return sampled_actions, norm_target_q, b_mu, b_A
    

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

        b_std = torch.diagonal(b_A, dim1=-2, dim2=-1)  # (B, da)
        std   = torch.diagonal(A,  dim1=-2, dim2=-1)  # (B, da)

        pi_1 = Independent(Normal(mu, b_std), 1)
        pi_2 = Independent(Normal(b_mu, std), 1)
        logp1 = pi_1.log_prob(sampled_actions)
        logp2 = pi_2.log_prob(sampled_actions)

        # Logging of variances
        var = std.pow(2)  # (B, da)
        var_mean = var.mean()     # Skalar
        var_min  = var.min()      # Skalar
        var_max  = var.max()      # Skalar

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

        # Logging statistics for analysis/monitoring
        self.mean_loss_p.append((-self.loss_p).item())
        self.max_kl_mu.append(C_mu_dim.max().item())
        self.max_kl_sigma.append(C_sigma_dim.max().item())
        self.min_kl_mu.append(C_mu_dim.min().item())
        self.min_kl_sigma.append(C_sigma_dim.min().item())
        self.var_mean.append(var_mean.item())
        self.var_min.append(var_min.item())
        self.var_max.append(var_max.item())
        self.mean_loss_l.append(self.loss_l.item())
    
    def critic_update_td(self, state_batch, action_batch, next_state_batch, reward_batch, sample_num=32):
        B = state_batch.size(0)
        sample_num = self.sample_action_num
        with torch.no_grad():

            ## get mean, cholesky from target actor --> to sample from Gaussian
            
            #!concatenation of all steps to perform only one forward pass of action sampling to reduce computional costs!
            all_states = torch.cat([state_batch, next_state_batch], dim=0)  # (2B, obs_dim)

            pi_mean, pi_A = self.target_actor.forward(all_states)  # (2B,)
            policy = MultivariateNormal(pi_mean, scale_tril=pi_A)  # (2B,)

            all_sampled_actions = policy.sample((sample_num,)).permute(1, 0, 2)  # (2B, sample_num, action_dim)
            sampled_actions      = all_sampled_actions[:B]  # (B, sample_num, action_dim)
            sampled_next_actions = all_sampled_actions[B:]  # (B, sample_num, action_dim)
            b_mu = pi_mean[:B]
            b_A =pi_A[:B]
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
        t = self.critic( state_batch, action_batch).squeeze()
        loss = self.norm_loss_q(y, t)
        loss.backward()
        self.critic_optimizer.step()
        return loss, t, y, sampled_actions_NBA, b_mu, b_A


    def train(self, log_callback =None):
        """
        Main training loop for MPO.
        - Collects experience from the environment.
        - Updates critic (policy evaluation).
        - Runs MPO E-step and M-step (policy improvement).
        - Periodically evaluates and logs statistics.
        """

        max_training_steps = self.max_training_steps
        num_steps = 0
        print(f"[DEBUG] start with number of steps = {len(self.replaybuffer)}, maximal number of environment steps: {self.max_training_steps}")       

        # Buffer size for warm up
        self.buffer_size = len(self.replaybuffer)
        
        # Precompute how many critic/actor updates we want per iteration
        # UTD_ratio * (num episodes * max steps per episode)
       
        # Warm-up: fill replay buffer with some initial experience
        while self.buffer_size < self.warm_up_steps:
            new_steps = self.sample_trajectory()
            num_steps += new_steps
            # Update buffer size after adding new episodes
            self.buffer_size= len(self.replaybuffer)
        
        # Main training iterations
        pbar = tqdm(total=max_training_steps, desc="Env steps")
        it = self.start_iteration
        while num_steps < max_training_steps:

            # Collect fresh experience for this iteration
            t_env_start = time.perf_counter()
            new_steps = self.sample_trajectory()

            #Update current steps for while loop
            num_steps += new_steps
            
            if self.wandb_track and it % self.log_videos_period == 0:
                prefix = f"rollout_gu{self.global_update}"
                self._log_one_episode_video(name_prefix=prefix)

            # For terminal logging
            pbar.update(new_steps)

            # Logging runtime env end
            t_env_end = time.perf_counter()
            self.runtime_env += t_env_end - t_env_start

            num_updates_per_iter = math.ceil(
            self.UTD_ratio * new_steps
            )

            # Perform several updates per iteration (UTD ratio)
            for r in range(num_updates_per_iter):
                
                self.buffer_size= len(self.replaybuffer)
                # Sample a minibatch from buffer    
                indices = np.random.choice(
                    self.buffer_size,
                    size=self.batch_size,
                    replace=False  # oder True, wenn du sehr viele Updates machen willst
                )        
                
                # Unpack transitions sampled from replay buffer    
                s_chunks, a_chunks, ns_chunks, r_chunks = self.replaybuffer.sample_batch(self.batch_size)

                state_batch      = torch.as_tensor(np.concatenate(s_chunks, axis=0), dtype=torch.float32, device=self.device)
                action_batch     = torch.as_tensor(np.concatenate(a_chunks, axis=0), dtype=torch.float32, device=self.device)
                next_state_batch = torch.as_tensor(np.concatenate(ns_chunks, axis=0), dtype=torch.float32, device=self.device)
                reward_batch     = torch.as_tensor(np.concatenate(r_chunks, axis=0), dtype=torch.float32, device=self.device)

                # Policy evaluation (critic update)
                t_policy_eval_start = time.perf_counter()

                if self.use_retrace:
                    loss_q, Q_current, Q_target = self.critic_update_retrace(state_batch, action_batch, next_state_batch, reward_batch, self.sample_action_num)
                else:
                    loss_q, Q_current, Q_target, sampled_actions, b_mu, b_A = self.critic_update_td( state_batch, action_batch, next_state_batch, reward_batch, self.sample_action_num)

                t_policy_eval_end = time.perf_counter()    
                self.runtime_policy_eval += t_policy_eval_end - t_policy_eval_start

                # Track critic statistics for logging
                self.mean_loss_q.append(loss_q.item())
                self.mean_current_q.append(Q_current.abs().mean().item())
                self.mean_target_q.append(Q_target.abs().mean().item())

                if r % self.delay_policy_update == 0:
                    # E-step (build non-parametric target distribution)
                    t_E_step_start = time.perf_counter()
                    sampled_actions, norm_target_q, b_mu, b_A = self.expectation_step(state_batch, sampled_actions, b_mu, b_A)
                    t_E_step_end = time.perf_counter()         
                    self.runtime_E_step += t_E_step_end - t_E_step_start

                    # M-step (actor / policy update)
                    t_M_step_start = time.perf_counter()
                    self.maximization_step(state_batch,norm_target_q, sampled_actions, b_mu, b_A)
                    t_M_step_end = time.perf_counter()
                    self.runtime_M_step += t_M_step_end - t_M_step_start

                # Inner-loop logging (e.g. every few gradient steps)
                if r % self.log_inner_interval == 0:
                    
                    # Compute mean reward/return in replay buffer
                    self.mean_reward_buffer = self.replaybuffer.mean_reward()
                    self.mean_return_buffer = self.replaybuffer.mean_return()                 

                    # Build log dict from current statistics    
                    logs = self._build_logs()
                    
                    # Send logs to WandB or other callback if enabled
                    if self.wandb_track is True and log_callback is not None:
                        log_callback(logs)

                    # Reset lists of per-update stats for the next logging window
                    self.reset_logs()

                # Target network update & logging
                # Periodically sync target networks with current actor/critic
                if self.global_update % self.target_update_period == 0:
                    self.update_target_actor_critic()

                # Increase global update counter after each gradient update    
                self.global_update += 1

            # Evaluation in the outer loop
            if it % self.evaluate_period == 0:
                
                t_eval_start = time.perf_counter()

                # Store current iteration index for logging
                self.iteration = it

                # Evaluate current policy without gradient tracking
                self.actor.eval()
                return_eval = self.evaluate()
                self.actor.train()

                # Build evaluation logs (separate from training logs)
                logs = {
                    "num_steps": self.num_steps,
                    "global_update": self.global_update,
                    "iteration": self.iteration,
                    "mean_return_buffer": self.mean_return_buffer,
                    "mean_reward_buffer": self.mean_reward_buffer,
                    "return_eval": return_eval,
                    "buffer_size": self.buffer_size
                    }
                if log_callback is not None:
                    log_callback(logs)

                t_eval_end = time.perf_counter()
                self.runtime_eval += t_eval_end -t_eval_start 
            
            self.save(it)
            it += 1  
        pbar.close()

    ## Mistake: need to change buffer for storage of old policies!
    def critic_update_retrace(self, state_batch, action_batch, next_state_batch, reward_batch, sample_num=64):
       
        with torch.no_grad():

            log_prob = self.actor.evaluate_action(state_batch, action_batch)
            target_log_prob = self.target_actor.evaluate_action(state_batch, action_batch)

            c_ret = self.calc_retrace_weights(target_log_prob, log_prob)

            with torch.no_grad():
                print("🔍 c_ret stats — min:", c_ret.min().item(), 
                    "max:", c_ret.max().item(), 
                    "mean:", c_ret.mean().item())

            Q_target = self.target_critic(state_batch, action_batch).squeeze(-1)
              # [B, N, act_dim]
            sampled_actions = self.target_actor.sample_action(next_state_batch, sample_num=sample_num)  # [B, N, act_dim]
            s_next_expanded = next_state_batch.unsqueeze(1).expand(-1, sample_num, -1)  # [B, N, obs_dim]
            
            flat_s = s_next_expanded.reshape(-1, next_state_batch.shape[-1])
            flat_a = sampled_actions.reshape(-1, action_batch.shape[-1])
            flat_q = self.target_critic.forward(flat_s, flat_a).squeeze(-1)  # [B*N]

            q_values = flat_q.view(state_batch.shape[0], sample_num)  # [B, N]
            expected_target_Q = q_values.mean(dim=1)  # [B]
            
            Q_ret = reward_batch + self.gamma * (expected_target_Q + c_ret * (expected_target_Q - Q_target))

        current_Q = self.critic(state_batch, action_batch).squeeze(-1)  # [B]
        
        critic_loss = self.norm_loss_q(current_Q, Q_ret)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        return critic_loss, current_Q, Q_ret

    def calc_retrace_weights(self,target_policy_logprob, behaviour_policy_logprob):
        
        log_retrace_weights = (target_policy_logprob - behaviour_policy_logprob).clamp(max=0)
        retrace_weights = log_retrace_weights.exp()
        assert not torch.isnan(log_retrace_weights).any(), "Error, a least one NaN value found in retrace weights."
        return retrace_weights
    
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

    
    def evaluate(self):
        """
        Run evaluation episodes using the current policy (self.actor)
        and return the average total reward per episode.
        """
        # No gradients needed during evaluation
        with torch.no_grad():
            total_rewards = []
            for ep_idx in range(self.evaluate_episode_num):
                total_reward = 0.0

                # Reset environment at the beginning of each episode
                state, info = self.env.reset()
                for s in range(self.evaluate_episode_maxstep):

                    # Convert state to tensor on the correct device
                    state_tensor = torch.as_tensor(
                        state, dtype=torch.float32, device=self.device
                        )
                    
                    # Get action from actor
                    action = self.actor.action(state_tensor, deterministic = True)

                    # Step environment
                    next_state, reward, terminated, truncated, info = self.env.step(action)
                    done = terminated or truncated

                    # Optional rendering (only first episode to avoid slowdown)
                    if self.render and ep_idx == 0:
                        self.env.render()
                        time.sleep(0.01)
                    
                    # Accumulate reward
                    total_reward += reward
                    if done:
                        break
                    state =  next_state
                total_rewards.append(total_reward)

        # Average return over all evaluation episodes
        return float(np.mean(total_rewards))



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



    def _log_one_episode_video(self, name_prefix: str):
        if self.make_video_env is None:
            return

        venv = self.make_video_env(name_prefix=name_prefix)
        try:
            # eine Episode deterministic laufen lassen
            self.actor.eval()
            state, _ = venv.reset()
            done = False
            steps = 0
            total_reward = 0.0
            with torch.no_grad():
                while not done and steps < self.evaluate_episode_maxstep:
                    st = torch.as_tensor(state, dtype=torch.float32, device=self.device)
                    action = self.actor.action(st, deterministic=True)
                    state, reward, terminated, truncated, _ = venv.step(action)
                    done = terminated or truncated
                    total_reward += float(reward)
                    steps += 1
            self.actor.train()

            
        finally:
            venv.close()

        video_folder = os.path.join(self.video_dir, self.run_name)  # videos/<run_name>
        mp4s = sorted(glob.glob(os.path.join(video_folder, "*.mp4")), key=os.path.getmtime)

        if not mp4s:
            return
        latest = mp4s[-1]

        wandb.log(
            {
                "rollout/video": wandb.Video(latest, format="mp4"),
                "rollout/video_return": total_reward,
                "rollout/video_len": steps,
            },
            step=self.global_update,
        )

        # optional: clean up
        os.remove(latest)


    def load_model(self, path=None):
        load_path = path if path is not None else self.save_path
        with torch.serialization.safe_globals([np.core.multiarray.scalar]):
            checkpoint = torch.load(load_path, weights_only=False)
  
       
        self.critic.load_state_dict(checkpoint['critic'])
        self.target_critic.load_state_dict(checkpoint['target_critic'])
        self.actor.load_state_dict(checkpoint['actor'])
        self.target_actor.load_state_dict(checkpoint['target_actor'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optim'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optim'])

        # load Lagrange multipliers
        self.eta = checkpoint["eta"]
        self.eta_mu = checkpoint["eta_mu"]
        self.eta_sigma = checkpoint["eta_sigma"]
                
        # replay buffer
        if "replay_buffer" in checkpoint:
            self.replaybuffer.load_state_dict(checkpoint["replay_buffer"])

        self.critic.train()
        self.target_critic.train()
        self.actor.train()
        self.target_actor.train()

    def save_lightweight(self, path, iteration):
        """Fast checkpoint: networks + optimizers only."""
        data = {
            "iteration": iteration,
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "target_actor": self.target_actor.state_dict(),
            "target_critic": self.target_critic.state_dict(),
            "actor_optim": self.actor_optimizer.state_dict(),
            "critic_optim": self.critic_optimizer.state_dict(),

            # Lagrange multipliers
            "eta": self.eta,
            "eta_mu": self.eta_mu,
            "eta_sigma": self.eta_sigma
        }
        torch.save(data, path)

    def save_full(self, path,iteration):
        """Full checkpoint including replay buffer."""
        data = {
            "iteration": iteration,
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "target_actor": self.target_actor.state_dict(),
            "target_critic": self.target_critic.state_dict(),
            "actor_optim": self.actor_optimizer.state_dict(),
            "critic_optim": self.critic_optimizer.state_dict(),

            # Lagrange multipliers
            "eta": self.eta,
            "eta_mu": self.eta_mu,
            "eta_sigma": self.eta_sigma,

            # Replay buffer
            "replay_buffer": self.replaybuffer.state_dict()
        }
        tmp = path + ".tmp"
        torch.save(data, tmp)
        os.replace(tmp, path)

    def save(self, iteration):
        """Main save function controlled by configuration."""
        # 1) ALWAYS save a lightweight latest model
        if self.save_latest:
            self.save_lightweight(
                os.path.join(self.model_dir, "latest_model_without_rb.pt"),
                iteration
            )

        # 2) EVERY N iterations save a full snapshot
        if iteration % self.save_every == 0:
            
            if self.save_replay_buffer:
                path = os.path.join(self.model_dir, "full_backup.pt")
                if os.path.exists(path):
                    os.remove(path)
                self.save_full(path,iteration)
            else:
                self.save_lightweight(os.path.join(self.model_dir, "latest_model_without_rb.pt"))

    def reset_logs(self):
        """
        Clear all accumulated statistics for the next logging window.
        These lists collect values over many gradient updates and are averaged
        inside _build_logs().
        """
        # Critic losses
        self.mean_loss_q.clear()
        self.mean_current_q.clear()
        self.mean_target_q.clear()

        # Actor losses
        self.mean_loss_p.clear()
        self.mean_loss_l.clear()

        # KL diagnostics
        self.max_kl_mu.clear()
        self.max_kl_sigma.clear()
        self.max_kl_mu.clear()
        self.max_kl_sigma.clear()

        # Covariance/variance diagnostics
        self.var_mean.clear()
        self.var_min.clear()
        self.var_max.clear()

    def _build_logs(self):
        """
        Build a dictionary with all training statistics for logging.
        Keeps the train() function clean.
        """
        eta_sigma_mean = float(self.eta_sigma.mean().item()) if torch.is_tensor(self.eta_sigma) else float(np.mean(self.eta_sigma))
        eta_sigma_max  = float(self.eta_sigma.max().item())  if torch.is_tensor(self.eta_sigma) else float(np.max(self.eta_sigma))
        eta_sigma_min  = float(self.eta_sigma.min().item())  if torch.is_tensor(self.eta_sigma) else float(np.min(self.eta_sigma))
        
        eta_mu_mean = float(self.eta_mu.mean().item()) if torch.is_tensor(self.eta_mu) else float(np.mean(self.eta_mu))
        eta_mu_max  = float(self.eta_mu.max().item())  if torch.is_tensor(self.eta_mu) else float(np.max(self.eta_mu))
        eta_mu_min  = float(self.eta_mu.min().item())  if torch.is_tensor(self.eta_mu) else float(np.min(self.eta_mu))


        return {
            "global_update": self.global_update,
            "mean_return_buffer": self.mean_return_buffer,
            "mean_reward_buffer": self.mean_reward_buffer,

            # Loss statistics (averaged since last reset)
            "mean_loss_q": np.mean(self.mean_loss_q) if self.mean_loss_q else 0.0,
            "mean_loss_p": np.mean(self.mean_loss_p) if self.mean_loss_p else 0.0,
            "mean_loss_l": np.mean(self.mean_loss_l) if self.mean_loss_l else 0.0,
            "mean_current_q": np.mean(self.mean_current_q) if self.mean_current_q else 0.0,
            "mean_target_q": np.mean(self.mean_target_q) if self.mean_target_q else 0.0,

            # Runtime statistics
            "runtime_env": self.runtime_env,
            "runtime_eval": self.runtime_eval,
            "runtime_policy_eval": self.runtime_policy_eval,
            "runtime_M_step": self.runtime_M_step,
            "runtime_E_step": self.runtime_E_step,

            # MPO-specific parameters
            "eta_mu_mean": eta_mu_mean,
            "eta_mu_max": eta_mu_max,
            "eta_mu_min": eta_mu_min,
            "eta_sigma_mean": eta_sigma_mean,
            "eta_sigma_max": eta_sigma_max,
            "eta_sigma_min": eta_sigma_min,

            # KL + variance diagnostics          
            "max_kl_mu": np.mean(self.max_kl_mu) if self.max_kl_mu else 0.0,
            "max_kl_sigma": np.mean(self.max_kl_sigma) if self.max_kl_sigma else 0.0,
            "min_kl_sigma": np.mean(self.min_kl_sigma) if self.min_kl_sigma else 0.0,
            "min_kl_mu": np.mean(self.min_kl_mu) if self.min_kl_mu else 0.0,
            "var_mean": np.mean(self.var_mean) if self.var_mean else 0.0,
            "var_min": np.mean(self.var_min) if self.var_min else 0.0,
            "var_max": np.mean(self.var_max) if self.var_max else 0.0,
        }
