import torch
import torch.nn.functional as F
from torch.distributions import Independent, Normal


def _flat_dim_from_env(env, which: str) -> int:
    if which == "obs":
        space = getattr(env, "single_observation_space", env.observation_space)
    elif which == "act":
        space = getattr(env, "single_action_space", env.action_space)
    else:
        raise ValueError(which)
    return int(np.prod(space.shape))

def compile_mpo_modules(mpo, *, mode="reduce-overhead", fullgraph=False, dynamic=False):
    if not hasattr(torch, "compile"):
        print("⚠️ torch.compile not available in this torch version. Skipping.")
        return mpo

    mpo.actor.forward        = torch.compile(mpo.actor.forward,        mode=mode, fullgraph=fullgraph, dynamic=dynamic)
    mpo.critic.forward       = torch.compile(mpo.critic.forward,       mode=mode, fullgraph=fullgraph, dynamic=dynamic)
    mpo.target_actor.forward = torch.compile(mpo.target_actor.forward, mode=mode, fullgraph=fullgraph, dynamic=dynamic)
    mpo.target_critic.forward= torch.compile(mpo.target_critic.forward,mode=mode, fullgraph=fullgraph, dynamic=dynamic)
    return mpo

@torch.no_grad()
def _rollout_sample(actor, obs_flat):
    mu, std = actor.forward(obs_flat)
    dist = Independent(Normal(mu, std), 1)
    return dist.sample()

def warmup_mpo_compile(args, device, env, mpo, *, compile_mode: str = "reduce-overhead"):
    print(f"🔥 torch.compile warmup on {device} ...")

    # use args dims (already set in make_envs), fallback to env dims
    obs_dim = int(getattr(args, "obs_space", _flat_dim_from_env(env, "obs")))
    act_dim = int(getattr(args, "action_dim", _flat_dim_from_env(env, "act")))

    B = int(args.batch_size)
    S = int(args.sample_action_num)
    gamma = float(getattr(args, "discount_factor", 0.99))

    mpo = compile_mpo_modules(mpo, mode=compile_mode, fullgraph=False, dynamic=False)

    # pick rollout batch sizes so vector env doesn't trigger recompiles later
    rollout_bs = {1, B}
    if isinstance(env, gym.vector.VectorEnv):
        rollout_bs.add(int(env.num_envs))
    rollout_bs = sorted(rollout_bs)

    # --- ZONE 1: ROLLOUT (no grad) ---
    mpo.actor.eval()
    mpo.target_actor.eval()
    for bs in rollout_bs:
        obs = torch.zeros((bs, obs_dim), device=device, dtype=torch.float32)
        _ = _rollout_sample(mpo.actor, obs)
        _ = _rollout_sample(mpo.target_actor, obs)
    print(f"   - Rollout warmup done (bs={rollout_bs}).")

    # --- ZONE 2: CRITIC BACKWARD ---
    mpo.actor.train()
    mpo.critic.train()
    mpo.target_actor.eval()
    mpo.target_critic.eval()

    state      = torch.zeros((B, obs_dim), device=device, dtype=torch.float32)
    next_state = torch.zeros((B, obs_dim), device=device, dtype=torch.float32)
    action     = torch.zeros((B, act_dim), device=device, dtype=torch.float32)
    reward     = torch.zeros((B,),         device=device, dtype=torch.float32)

    with torch.no_grad():
        all_states = torch.cat([state, next_state], dim=0)          # (2B, obs)
        mu_all, std_all = mpo.target_actor.forward(all_states)      # mu_all: (2B, act), std_all: (act) OR (2B, act)

        pi_all = Independent(Normal(mu_all, std_all), 1)
        all_sampled_actions = pi_all.sample((S,))                   # (S, 2B, act)

        all_states_exp = all_states.unsqueeze(0).expand(S, -1, -1)  # (S, 2B, obs)
        all_q = mpo.target_critic.forward(
            all_states_exp.reshape(S * 2 * B, obs_dim),
            all_sampled_actions.reshape(S * 2 * B, act_dim),
        ).view(S, 2 * B)

        expected_next_q = all_q[:, B:].mean(dim=0)                  # (B,)
        y = reward + gamma * expected_next_q                        # (B,)

    q = mpo.critic.forward(state, action).view(-1)                  # (B,)
    critic_loss = F.mse_loss(q, y)
    critic_loss.backward()
    mpo.critic.zero_grad(set_to_none=True)
    print("   - Critic backward warmup done.")

    # --- ZONE 3: ACTOR BACKWARD (M-step-like) ---
    with torch.no_grad():
        all_states = torch.cat([state, next_state], dim=0)          # (2B, obs)
        mu_all, std_all = mpo.target_actor.forward(all_states)

        pi_all = Independent(Normal(mu_all, std_all), 1)
        all_sampled_actions = pi_all.sample((S,))                   # (S, 2B, act)

        all_states_exp = all_states.unsqueeze(0).expand(S, -1, -1)
        all_q = mpo.target_critic.forward(
            all_states_exp.reshape(S * 2 * B, obs_dim),
            all_sampled_actions.reshape(S * 2 * B, act_dim),
        ).view(S, 2 * B)

        q_sa = all_q[:, :B]                                         # (S, B)

        eta_val = float(getattr(mpo, "eta_dual", torch.tensor(1.0)).detach().cpu().item())
        eta_val = max(eta_val, 1e-6)
        weights = torch.softmax(q_sa / eta_val, dim=0)              # (S, B)

        sampled_actions = all_sampled_actions[:, :B, :]             # (S, B, act)
        mu_b  = mu_all[:B, :]                                       # (B, act)

        # std can be (act,) OR (2B,act). We just need "old std for first B".
        std_b = std_all if std_all.ndim == 1 else std_all[:B, :]     # (act,) or (B,act)

    mu, std = mpo.actor.forward(state)                               # mu: (B,act), std: (act,) or (B,act)

    pi1 = Independent(Normal(mu,   std_b), 1)                        # broadcast ok
    pi2 = Independent(Normal(mu_b, std),   1)

    logp1 = pi1.log_prob(sampled_actions)                            # (S, B)
    logp2 = pi2.log_prob(sampled_actions)                            # (S, B)

    actor_loss = -(weights * (logp1 + logp2)).mean()
    actor_loss.backward()
    mpo.actor.zero_grad(set_to_none=True)
    print("   - Actor backward warmup done.")

    print("✅ torch.compile warmup complete.")
    return mpo