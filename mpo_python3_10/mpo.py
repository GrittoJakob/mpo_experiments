
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


def btr(m):
    return m.diagonal(dim1=-2, dim2=-1).sum(-1)

def bt(m):
    return m.transpose(dim0=-2, dim1=-1)

def gaussian_kl(mu_i, mu, Ai, A):
    """
       Computes KL for multivariate Gaussians with Cholesky factors Ai (old) and A (new).
    Returns:
      - C_mu:    mean KL for mean
      - C_sigma: mean KL for covariance
      - sigma_i_det: mean determinant of old cov
      - sigma_det: mean determinant of new cov
      - var_mean: mean diagonal variance of Σ
      - var_min:  minimum diagonal variance
      - var_max:  maximum diagonal variancen: mean of determinanats of sigma_i, sigma

    """
    n = A.size(-1)
    mu_i = mu_i.unsqueeze(-1)  # (B, n, 1)
    mu = mu.unsqueeze(-1)  # (B, n, 1)

    sigma_i = Ai @ bt(Ai)  # (B, n, n)
    sigma = A @ bt(A)  # (B, n, n)
    sigma_i_det = sigma_i.det()  # (B,)
    sigma_det = sigma.det()  # (B,)
    sigma_i_det = torch.clamp_min(sigma_i_det, 1e-6)
    sigma_det = torch.clamp_min(sigma_det, 1e-6)
    sigma_i_inv = sigma_i.inverse()  # (B, n, n)
    sigma_inv = sigma.inverse()  # (B, n, n)

    inner_mu = ((mu - mu_i).transpose(-2, -1) @ sigma_i_inv @ (mu - mu_i)).squeeze()  # (B,)
    inner_sigma = torch.log(sigma_det / sigma_i_det) - n + btr(sigma_inv @ sigma_i)  # (B,)
    C_mu = 0.5 * torch.mean(inner_mu)
    C_sigma = 0.5 * torch.mean(inner_sigma)
    
    var_diag = torch.diagonal(sigma, dim1=-2, dim2=-1)  # (B,n)
    var_mean = var_diag.mean()         # Mean 
    var_min  = var_diag.min()          # Minimal variance
    var_max  = var_diag.max()          # maximum variance
    return C_mu, C_sigma, torch.mean(sigma_i_det), torch.mean(sigma_det), var_mean, var_min, var_max
        
     

class MPO(object):
    def __init__(self, env, args):
        self.env = env
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]

        self.device = args.device
        self.eps_daul = args.dual_constraint
        self.eps_mu = args.kl_mean_constraint
        self.eps_gamma = args.kl_var_constraint
        self.gamma = args.discount_factor
        self.alpha_mu_scale = args.alpha_mean_scale # scale Largrangian multiplier
        self.alpha_sigma_scale = args.alpha_var_scale  # scale Largrangian multiplier
        self.alpha_mu_max = args.alpha_mean_max
        self.alpha_sigma_max = args.alpha_var_max

        self.sample_episode_num = args.sample_episode_num
        self.sample_episode_maxstep = args.sample_episode_maxstep
        self.sample_action_num = args.sample_action_num
        self.batch_size = args.batch_size
        self.num_updates_per_iter = args.num_updates_per_iter
        self.mstep_iteration_num = args.mstep_iteration_num
        self.evaluate_period = args.evaluate_period
        self.evaluate_episode_num = args.evaluate_episode_num
        self.evaluate_episode_maxstep = args.evaluate_episode_maxstep
        self.learning_rate = args.learning_rate
        self.eta_lr = args.eta_lr
        self.clear_replay_buffer = args.clear_replay_buffer
        self.save_every = args.save_every
        self.save_latest = args.save_latest
        self.wandb_track = args.track
        self.std_init = args.std_init
        self.use_tanh_mean = args.use_tanh_mean
        self.use_retrace = args.use_retrace

        self.actor = Actor(env, args.hidden_size_actor, self.std_init, self.use_tanh_mean).to(self.device)
        self.critic = Critic(env, args.hidden_size_critic).to(self.device)
        self.target_actor = Actor(env,args.hidden_size_actor, self.std_init, self.use_tanh_mean).to(self.device)
        self.target_critic = Critic(env,args.hidden_size_critic).to(self.device)
        self.save_replay_buffer = args.save_replay_buffer
        self.q_update_step = 0
        self.target_update_period = args.target_update_period
        
    

        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)
            target_param.requires_grad = False
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)
            target_param.requires_grad = False

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),self.learning_rate)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), self.learning_rate)
        self.norm_loss_q = nn.MSELoss() if args.q_loss_type == 'mse' else nn.SmoothL1Loss()

        self.replaybuffer = ReplayBuffer()
        self.log_dir = args.log_dir
        self.model_dir = os.path.join(self.log_dir, "model")
        os.makedirs(self.model_dir, exist_ok=True)


        self.eta = np.random.rand()
        self.eta_mu = 0.0  # lagrangian multiplier for continuous action space in the M-step
        self.eta_sigma = 0.0  # lagrangian multiplier for continuous action space in the M-step
        self.num_steps = 0
        self.max_return_eval = -np.inf
        self.start_iteration = 1
        self.render = False
        self.update_dual_function_interval = args.update_dual_function_interval
        self.log_inner_interval = args.log_inner_interval 

        self.mean_loss_q = []
        self.mean_loss_p = []
        self.mean_loss_l = []
        self.mean_current_q = []
        self.mean_target_q = []
        self.max_kl_mu = []
        self.max_kl_sigma = []
        self.mean_sigma_det = []
        self.max_kl = []
        self.var_mean = []
        self.var_min = []
        self.var_max = []

    def sample_trajectory(self, sample_episode_num):
        if self.clear_replay_buffer:
            self.replaybuffer.clear()
        episodes = []

        for ep_idx in range(self.sample_episode_num):
            buff = []
            state, info = self.env.reset()

            for steps in range(self.sample_episode_maxstep):
                if not np.isfinite(state).all():
                    print("⚠️ NaN/Inf in state:", state)
                    raise ValueError("State contains NaN/Inf before Actor!")
                state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device)
                #print(state.shape, state_tensor.shape)
                action = self.target_actor.action(state_tensor).cpu().numpy()

                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated  # Gymnasium: done = terminated OR truncated

                buff.append((state, action, next_state, reward))

                if self.render and epd_idx == 0:
                    # bei Ant-v5: render_mode="human" schon beim gym.make(...) setzen
                    self.env.render()
                    sleep(0.01)

                if done:
                    break
                state = next_state

            episodes.append(buff)
        self.replaybuffer.store_episodes(episodes)


    def train(self, iteration_num=None, render=None, log_callback =None):
        print(f"[DEBUG] start_iteration = {self.start_iteration}, iteration_num = {iteration_num}")
        self.render = render
        global_update = 1
        all_logs = []
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
                        
                    K = self.batch_size  # the sample number of states
                    N = self.sample_action_num  # the sample number of actions per state

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
                        target_q_np = target_q.cpu().transpose(0, 1).numpy()  # (K, N)
                        
                    
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
                
            

        return all_logs



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


    def update_target_actor_critic(self):
        # param(target_actor) <-- param(actor)
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)

        # param(target_critic) <-- param(critic)
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)
    
    def dual_function(self, eta, target_q, eps_daul):
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
            eta * eps_daul
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

    def evaluate(self):
        with torch.no_grad():
            total_rewards = []
            for e in range(self.evaluate_episode_num):
                total_reward = 0.0
                state, info = self.env.reset()
                for s in range(self.evaluate_episode_maxstep):
                    action = self.actor.action(torch.as_tensor(state, dtype=torch.float32, device=self.device)).cpu().numpy()
                    next_state, reward, terminated, truncated, info = self.env.step(action)
                    done = terminated or truncated
                    total_reward += reward
                    if done:
                        break
                    state =  next_state
                total_rewards.append(total_reward)
            return np.mean(total_rewards)

    def reset_logs(self):
      
        self.mean_loss_q.clear()
        self.mean_loss_p.clear()
        self.mean_loss_l.clear()
        self.mean_current_q = []
        self.mean_target_q = []
        self.max_kl_mu.clear()
        self.max_kl_sigma.clear()
        self.mean_sigma_det.clear()