
import os
from time import sleep
import numpy as np
from scipy.optimize import minimize
from tqdm import tqdm
import torch
import gymnasium as gym
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.distributions import MultivariateNormal
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from actor import Actor
from critic import Critic
from replaybuffer import ReplayBuffer
import time
import math
from utils import btr,  bt, gaussian_kl
        
class MPO(object):
    def __init__(self, env, args):
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
        self.eps_dual = args.dual_constraint      # KL constraint for E-Step
        self.eps_mu = args.kl_mean_constraint     # KL constraint for mean
        self.eps_gamma = args.kl_var_constraint   # KL constraint for variance

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
    
        self.sample_episode_num = args.sample_episode_num    # Number of episodes to sample per iteration       
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

        # Main actor / critic networks
        self.actor = Actor(env, args.hidden_size_actor, self.std_init).to(self.device)
        self.critic = Critic(env, args.hidden_size_critic).to(self.device)

        # Target networks (used for stable targets)
        self.target_actor = Actor(env, args.hidden_size_actor, self.std_init).to(self.device)
        self.target_critic = Critic(env, args.hidden_size_critic).to(self.device)

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
        self.eta_mu = 0.0      # for mean KL constraint
        self.eta_sigma = 0.0   # for variance KL constraint

        # Global step / iteration bookkeeping
        self.num_steps = 0
        self.start_iteration = 1
        self.global_update = 1
        
        # Rendering flag for evaluation (not used during training rollouts)
        self.render = args.render

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
        self.mean_sigma_det = []

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
        # Loop over episodes to collect
        for _ in range(self.sample_episode_num):
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
                done = terminated or truncated  # Gymnasium: done = terminated OR truncated
                
                # Store transition in the current episode buffer
                buff.append((state, action, next_state, reward))

                if done:
                    break
                state = next_state

            # Store completed episode
            episodes.append(buff)

        # Push all collected episodes into the replay buffer and returns the number of collected steps
        self.replaybuffer.store_episodes(episodes)
        return total_steps_collected

    def expectation_step(self, state_batch):
        """
        E-step of MPO:
        - Sample actions from the target policy for each state.
        - Evaluate Q-values with the target critic.
        - Update the temperature parameter eta via the dual function.
        - Compute normalized weights over actions (norm_target_q).
        """

        K = self.batch_size  # the sample number of states
        N = self.sample_action_num  # the sample number of actions per state

        with torch.no_grad():
            # Get mean and Cholesky factor of the target policy for each state
            b_mu, b_A = self.target_actor.forward(state_batch)  # (K,)

            # Gaussian policy over actions with batch size K
            b = MultivariateNormal(b_mu, scale_tril=b_A)  # (K,)

            # Sample N actions per state -> shape (N, K, action_dim)
            sampled_actions = b.sample((N,))  
            expanded_states = state_batch[None, ...].expand(N, -1, -1)  # (N, K, state_dim)

            # Evaluate Q-values of the target critic for all (state, action) pairs
            target_q = self.target_critic.forward(
                expanded_states.reshape(-1, self.state_dim), sampled_actions.reshape(-1, self.action_dim)  # (N * K, action_dim)
            ).reshape(N, K)
        
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
        # Shape stays (N, K), softmax along action-sample dimension N
        norm_target_q = torch.softmax(target_q / self.eta, dim=0)  # (N, K) or (action_dim, K)

        return sampled_actions, norm_target_q, b_mu, b_A
    

    def maximization_step(self, state_batch, norm_target_q, sampled_actions, b_mu, b_A): 
        """
        M-step of MPO:
        - Update the policy parameters to maximize the weighted log-likelihood
        under the non-parametric target distribution (defined by norm_target_q).
        - Enforce KL constraints between the new and old policy via Lagrange multipliers.
        """
        # sampled_actions has shape (N, K, action_dim)
        N, K, _ = sampled_actions.shape

        # Current actor parameters for this batch of states
        mu, A = self.actor.forward(state_batch)

        # paper1 version
        #policy = MultivariateNormal(loc=mu, scale_tril=A)  # (K,)
        #loss_p = torch.mean( norm_target_q * policy.expand((N, K)).log_prob(sampled_actions))  # (N, K)
        #C_mu, C_sigma, sigma_i_det, sigma_det = gaussian_kl( mu_i=b_mu, mu=mu, Ai=b_A, A=A)

        # Two Gaussian policies as in the "paper2" variant:
        # pi_1 uses new mean and old covariance; pi_2 uses old mean and new covariance.
        pi_1 = MultivariateNormal(loc=mu, scale_tril=b_A)  # (K,)
        pi_2 = MultivariateNormal(loc=b_mu, scale_tril=A)  # (K,)

        # Weighted policy improvement objective
        self.loss_p = torch.mean(
            norm_target_q * (
                pi_1.expand((N, K)).log_prob(sampled_actions)  # (N, K)
                + pi_2.expand((N, K)).log_prob(sampled_actions)  # (N, K)
            )
        )

        # KL constraints between old and new Gaussian policies
        C_mu, C_sigma, _, sigma_det,var_mean, var_min, var_max  = gaussian_kl( mu_i=b_mu, mu=mu, Ai=b_A, A=A)

        # Update lagrange multipliers by gradient descent
        self.eta_mu -= self.alpha_mu_scale * (self.eps_mu - C_mu).detach().item()
        self.eta_sigma -= self.alpha_sigma_scale * (self.eps_gamma - C_sigma).detach().item()

        # Clamp Lagrange multipliers to a reasonable range
        self.eta_mu = np.clip(self.eta_mu, 1e-10, self.alpha_mu_max)
        self.eta_sigma = np.clip(self.eta_sigma, 1e-10, self.alpha_sigma_max)

        # Total actor loss: maximize weighted log-prob subject to KL constraints
        self.actor_optimizer.zero_grad()
        self.loss_l = -( self.loss_p + self.eta_mu * (self.eps_mu - C_mu) + self.eta_sigma * (self.eps_gamma - C_sigma))
        self.loss_l.backward()
        clip_grad_norm_(self.actor.parameters(), 0.1)
        self.actor_optimizer.step() 

        # Logging statistics for analysis/monitoring
        self.mean_loss_p.append((-self.loss_p).item())
        self.max_kl_mu.append(C_mu.item())
        self.max_kl_sigma.append(C_sigma.item())
        self.mean_sigma_det.append(sigma_det.item())
        self.var_mean.append(var_mean.item())
        self.var_min.append(var_min.item())
        self.var_max.append(var_max.item())
        self.mean_loss_l.append(self.loss_l.item())


    def train(self, iteration_num=None, render=None, log_callback =None):
        """
        Main training loop for MPO.
        - Collects experience from the environment.
        - Updates critic (policy evaluation).
        - Runs MPO E-step and M-step (policy improvement).
        - Periodically evaluates and logs statistics.
        """
        print(f"[DEBUG] start_iteration = {self.start_iteration}, iteration_num = {iteration_num}")
        
        self.render = render

        # Buffer size for warm up
        self.buffer_size = len(self.replaybuffer)
        
        # Precompute how many critic/actor updates we want per iteration
        # UTD_ratio * (num episodes * max steps per episode)
       
        # Warm-up: fill replay buffer with some initial experience
        while self.buffer_size < self.warm_up_steps:
            _ = self.sample_trajectory()

            # Update buffer size after adding new episodes
            self.buffer_size= len(self.replaybuffer)
        
        # Main training iterations
        for it in tqdm(range(self.start_iteration, iteration_num + 1), desc="Training iterations"):

            # Collect fresh experience for this iteration
            t_env_start = time.perf_counter()
            steps = self.sample_trajectory()
            t_env_end = time.perf_counter()
            self.runtime_env += t_env_end - t_env_start

            num_updates_per_iter = math.ceil(
            self.UTD_ratio * steps
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
                state_batch, action_batch, next_state_batch, reward_batch = zip(
                    *[self.replaybuffer[index] for index in indices])

                # Convert to torch tensors on the correct device
                state_batch      = torch.as_tensor(np.stack(state_batch), dtype=torch.float32, device=self.device)  # [K, dim_obs]
                action_batch     = torch.as_tensor(np.stack(action_batch), dtype=torch.float32, device=self.device) # [K, dim_act]
                next_state_batch = torch.as_tensor(np.stack(next_state_batch), dtype=torch.float32, device=self.device) #[K, dim_obs]
                reward_batch     = torch.as_tensor(np.stack(reward_batch), dtype=torch.float32, device=self.device)     #[K,]
                
                # Policy evaluation (critic update)
                t_policy_eval_start = time.perf_counter()

                if self.use_retrace:
                    loss_q, Q_current, Q_target = self.critic_update_retrace(state_batch, action_batch, next_state_batch, reward_batch, self.sample_action_num)
                else:
                    loss_q, Q_current, Q_target = self.critic_update_td( state_batch, action_batch, next_state_batch, reward_batch, self.sample_action_num)

                t_policy_eval_end = time.perf_counter()    
                self.runtime_policy_eval += t_policy_eval_end - t_policy_eval_start

                # Track critic statistics for logging
                self.mean_loss_q.append(loss_q.item())
                self.mean_current_q.append(Q_current.abs().mean().item())
                self.mean_target_q.append(Q_target.abs().mean().item())

                if r % self.delay_policy_update == 0:
                    # E-step (build non-parametric target distribution)
                    t_E_step_start = time.perf_counter()
                    sampled_actions, norm_target_q, b_mu, b_A = self.expectation_step(state_batch)
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

            self.save(it)
            t_eval_end = time.perf_counter()
            self.runtime_eval += t_eval_end -t_eval_start     

    def critic_update_td(self, state_batch, action_batch, next_state_batch, reward_batch, sample_num=64):
        B = state_batch.size(0)
        with torch.no_grad():

            ## get mean, cholesky from target actor --> to sample from Gaussian
            pi_mean, pi_A = self.target_actor.forward(next_state_batch)  # (B,)
            policy = MultivariateNormal(pi_mean, scale_tril=pi_A)  # (B,)
            sampled_next_actions = policy.sample((sample_num,)).transpose(0, 1)  # (B, sample_num, action_dim)
            expanded_next_states = next_state_batch[:, None, :].expand(-1, sample_num, -1)  # (B, sample_num, state_dim)
            
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
        return loss, t, y

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


    def load_model(self, path=None):
        load_path = path if path is not None else self.save_path
        with torch.serialization.safe_globals([np.core.multiarray.scalar]):
            checkpoint = torch.load(load_path, weights_only=False)
  
        self.start_iteration = checkpoint['iteration'] + 1
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

        # Covariance/variance diagnostics
        self.mean_sigma_det.clear()
        self.var_mean.clear()
        self.var_min.clear()
        self.var_max.clear()

    def _build_logs(self):
        """
        Build a dictionary with all training statistics for logging.
        Keeps the train() function clean.
        """
        return {
            "num_steps": self.num_steps,
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
            "eta": self.eta,
            "eta_mu": self.eta_mu,
            "eta_sigma": self.eta_sigma,

            # KL + variance diagnostics
            "max_kl_mu": np.mean(self.max_kl_mu) if self.max_kl_mu else 0.0,
            "max_kl_sigma": np.mean(self.max_kl_sigma) if self.max_kl_sigma else 0.0,
            "mean_sigma_det": np.mean(self.mean_sigma_det) if self.mean_sigma_det else 0.0,
            "var_mean": np.mean(self.var_mean) if self.var_mean else 0.0,
            "var_min": np.mean(self.var_min) if self.var_min else 0.0,
            "var_max": np.mean(self.var_max) if self.var_max else 0.0,
        }

"""
!!! OLD training function !!!

def train(self, iteration_num=None, render=None, log_callback =None):
        print(f"[DEBUG] start_iteration = {self.start_iteration}, iteration_num = {iteration_num}")
        self.render = render
        global_update = 1
        env_runtime = 0
        train_runtime = 0
        eval_runtime = 0
        runtime_policy_eval = 0
        runtime_E_step = 0
        runtime_M_step = 0
        #writer = SummaryWriter(os.path.join(log_dir, 'tb'))

        for it in tqdm(range(self.start_iteration, iteration_num + 1), desc="Training iterations") :
            
            t_env_start = time.perf_counter()
            self.sample_trajectory(self.sample_episode_num)
            t_env_end = time.perf_counter()
            env_runtime += t_env_end - t_env_start

            t_train_start = time.perf_counter()
            buffer_size = len(self.replaybuffer)

            mean_reward_buffer = self.replaybuffer.mean_reward()
            mean_return_buffer = self.replaybuffer.mean_return()

            if buffer_size < self.batch_size:

                print(f"[MPO] Buffer warmup: {buffer_size} < batch_size={self.batch_size}, skip updates")

            else:

                for r in range(self.num_updates_per_iter):
                    
                    indices = np.random.choice(
                        buffer_size,
                        size=self.batch_size,
                        replace=False  # oder True, wenn du sehr viele Updates machen willst
                    )
                        
           

                    state_batch, action_batch, next_state_batch, reward_batch = zip(
                        *[self.replaybuffer[index] for index in indices])

                    state_batch      = torch.as_tensor(np.stack(state_batch), dtype=torch.float32, device=self.device)  # [K, dim_obs]
                    action_batch     = torch.as_tensor(np.stack(action_batch), dtype=torch.float32, device=self.device) # [K, dim_act]
                    next_state_batch = torch.as_tensor(np.stack(next_state_batch), dtype=torch.float32, device=self.device) #[K, dim_obs]
                    reward_batch     = torch.as_tensor(np.stack(reward_batch), dtype=torch.float32, device=self.device)     #[K,]

                    t_policy_eval_start = time.perf_counter()
                    # Policy Evaluation
                    if self.use_retrace:
                        loss_q, current_q, Q_target = self.critic_update_retrace(state_batch, action_batch, next_state_batch, reward_batch, self.sample_action_num)
                    else:
                        loss_q, current_q, Q_target = self.critic_update_td( state_batch, action_batch, next_state_batch, reward_batch, self.sample_action_num)

                    t_policy_eval_end = time.perf_counter()    
                    runtime_policy_eval += t_policy_eval_end - t_policy_eval_start
                    self.q_update_step += 1
                    self.mean_loss_q.append(loss_q.item())
                    self.mean_current_q.append(current_q.abs().mean().item())
                    self.mean_target_q.append(Q_target.abs().mean().item())
                    
                    t_E_step_start = time.perf_counter()        #Timer
                    # E-Step of Policy Improvement
                    with torch.no_grad():
                        # sample N actions per state
                        b_mu, b_A = self.target_actor.forward(state_batch)  # (K,)
                        b = MultivariateNormal(b_mu, scale_tril=b_A)  # (K,)
                        sampled_actions = b.sample((N,))  # (N, K, action_dim)
                        expanded_states = state_batch[None, ...].expand(N, -1, -1)  # (N, K, state_dim)
                        target_q = self.target_critic.forward(
                            expanded_states.reshape(-1, self.state_dim), sampled_actions.reshape(-1, self.action_dim)  # (N * K, action_dim)
                        ).reshape(N, K)
                        target_q = target_q.cpu().transpose(0, 1).numpy()  # (K, N)
                        
                    
                    if r % self.update_dual_function_interval == 0:  
                        eta_tensor = torch.tensor(self.eta, device=self.device, requires_grad=True)
                        dual_val = self.dual_function(eta_tensor, target_q.detach(), self.eps_daul)
                        dual_val.backward()
                        with torch.no_grad():
                            eta_tensor -= self.eta_lr * eta_tensor.grad
                            eta_tensor.clamp_(min=1e-6)
                            self.eta = eta_tensor.item()
                    # normalize
                    norm_target_q = torch.softmax(target_q / self.eta, dim=0)  # (N, K) or (action_dim, K)

                    t_E_step_end = time.perf_counter()         #Counter
                    runtime_E_step += t_E_step_end - t_E_step_start

                    t_M_step_start = time.perf_counter() 
                    # M-Step of Policy Improvement
                    for _ in range(self.mstep_iteration_num):
                        mu, A = self.actor.forward(state_batch)

                        # paper1 version
                        #policy = MultivariateNormal(loc=mu, scale_tril=A)  # (K,)
                        #loss_p = torch.mean( norm_target_q * policy.expand((N, K)).log_prob(sampled_actions))  # (N, K)
                        #C_mu, C_sigma, sigma_i_det, sigma_det = gaussian_kl( mu_i=b_mu, mu=mu, Ai=b_A, A=A)

                        # paper2 version normalize
                        pi_1 = MultivariateNormal(loc=mu, scale_tril=b_A)  # (K,)
                        pi_2 = MultivariateNormal(loc=b_mu, scale_tril=A)  # (K,)
                        self.loss_p = torch.mean(
                            norm_target_q * (
                                pi_1.expand((N, K)).log_prob(sampled_actions)  # (N, K)
                                + pi_2.expand((N, K)).log_prob(sampled_actions)  # (N, K)
                            )
                        )
                        C_mu, C_sigma, sigma_i_det, sigma_det,var_mean, var_min, var_max  = gaussian_kl( mu_i=b_mu, mu=mu, Ai=b_A, A=A)
                        
                        self.mean_loss_p.append((-self.loss_p).item())
                        self.max_kl_mu.append(C_mu.item())
                        self.max_kl_sigma.append(C_sigma.item())
                        self.mean_sigma_det.append(sigma_det.item())
                        self.var_mean.append(var_mean.item())
                        self.var_min.append(var_min.item())
                        self.var_max.append(var_max.item())

                        # Update lagrange multipliers by gradient descent
                    
                        self.eta_mu -= self.alpha_mu_scale * (self.eps_mu - C_mu).detach().item()
                        self.eta_sigma -= self.alpha_sigma_scale * (self.eps_gamma - C_sigma).detach().item()

                        self.eta_mu = np.clip(self.eta_mu, 1e-10, self.alpha_mu_max)
                        self.eta_sigma = np.clip(self.eta_sigma, 1e-10, self.alpha_sigma_max)

                        self.actor_optimizer.zero_grad()
                        self.loss_l = -( self.loss_p + self.eta_mu * (self.eps_mu - C_mu) + self.eta_sigma * (self.eps_gamma - C_sigma))
                        self.mean_loss_l.append(self.loss_l.item())
                        self.loss_l.backward()
                        clip_grad_norm_(self.actor.parameters(), 0.1)
                        self.actor_optimizer.step() 
                    
                    t_M_step_end = time.perf_counter()
                    runtime_M_step += t_M_step_end - t_M_step_start

                    if global_update % self.target_update_period == 0:
                        self.update_target_actor_critic()
                    #print("Debug:Update Targets")

                    if r % self.log_inner_interval == 0:
                        
                        self.num_steps = it * self.sample_episode_num* self.sample_episode_maxstep
                        
                        logs = {
                            "num_steps": self.num_steps,
                            "global_update": global_update,
                            "mean_return_buffer": mean_return_buffer,
                            "mean_reward_buffer": mean_reward_buffer,
                            "mean_loss_q": np.mean(self.mean_loss_q),
                            "mean_loss_p": np.mean(self.mean_loss_p),
                            "mean_loss_l": np.mean(self.mean_loss_l),
                            "mean_current_q": np.mean(self.mean_current_q),
                            "mean_target_q": np.mean(self.mean_target_q),
                            "runtime_train": train_runtime,
                            "runtime_env": env_runtime,
                            "runtime_eval": eval_runtime,
                            "runtime_policy_eval": runtime_policy_eval,
                            "runtime_M_step": runtime_M_step,
                            "runtime_E_step": runtime_E_step,
                            "eta": self.eta,
                            "max_kl_mu": np.mean(self.max_kl_mu),
                            "max_kl_sigma": np.mean(self.max_kl_sigma),
                            "mean_sigma_det": np.mean(self.mean_sigma_det),
                            "eta_mu": self.eta_mu,
                            "eta_sigma": self.eta_sigma,
                            "var_mean": np.mean(self.var_mean),
                            "var_min": np.mean(self.var_min),
                            "var_max": np.mean(self.var_max)
                        }
                        
                        if self.wandb_track is True and log_callback is not None:
                            log_callback(logs)
                        self.reset_logs()
                        
                    global_update += 1

                t_train_end = time.perf_counter()
                train_runtime += t_train_end - t_train_start
                
                #Evalutation in outer loop
                if it % self.evaluate_period == 0:

                    t_eval_start = time.perf_counter()
                    self.actor.eval()
                    return_eval = self.evaluate()
                    self.actor.train()
                    self.num_steps = it * self.sample_episode_num * self.sample_episode_maxstep
                    logs = {
                        "num_steps": self.num_steps,
                        "global_update": global_update,
                        "iteration": it,
                        "mean_return_buffer": mean_return_buffer,
                        "mean_reward_buffer": mean_reward_buffer,
                        "return_eval": return_eval,
                        }
                    if log_callback is not None:
                        log_callback(logs)

                self.save(it)
                t_eval_end = time.perf_counter()
                eval_runtime += t_eval_end -t_eval_start  
                

"""