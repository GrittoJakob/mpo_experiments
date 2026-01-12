import torch
import torch.nn.functional as F
from torch.distributions import Independent, Normal


def _space_shape(env, which: str):
    if which == "obs":
        space = getattr(env, "single_observation_space", env.observation_space)
    elif which == "act":
        space = getattr(env, "single_action_space", env.action_space)
    else:
        raise ValueError(which)
    return space.shape


def compile_mpo_modules(
    mpo,
    *,
    mode: str = "reduce-overhead",
    fullgraph: bool = False,
    dynamic: bool = False,
):
    if not hasattr(torch, "compile"):
        print("⚠️ torch.compile not available in this torch version. Skipping.")
        return mpo

    mpo.actor.forward = torch.compile(mpo.actor.forward, mode=mode, fullgraph=fullgraph, dynamic=dynamic)
    mpo.critic.forward = torch.compile(mpo.critic.forward, mode=mode, fullgraph=fullgraph, dynamic=dynamic)
    mpo.target_actor.forward = torch.compile(mpo.target_actor.forward, mode=mode, fullgraph=fullgraph, dynamic=dynamic)
    mpo.target_critic.forward = torch.compile(mpo.target_critic.forward, mode=mode, fullgraph=fullgraph, dynamic=dynamic)
    return mpo


@torch.no_grad()
def _rollout_kernel(actor, obs):
    mu, std = actor.forward(obs)
    dist = Independent(Normal(mu, std), 1)
    return dist.sample()


def warmup_mpo_compile(args, device, env, mpo, *, compile_mode: str = "reduce-overhead"):
    print(f"🔥 torch.compile warmup on {device} ...")

    obs_shape = _space_shape(env, "obs")
    act_shape = _space_shape(env, "act")

    B = int(args.batch_size)
    S = int(args.sample_action_num)

    mpo = compile_mpo_modules(mpo, mode=compile_mode, fullgraph=False, dynamic=False)

    # --- ZONE 1: ROLLOUT (no grad) ---
    for bs in (1, 1):
        obs = torch.zeros((bs, *obs_shape), device=device, dtype=torch.float32)
        _ = _rollout_kernel(mpo.actor, obs)
        _ = _rollout_kernel(mpo.target_actor, obs)
    print("   - Rollout forward warmup done.")

    # --- ZONE 2: CRITIC (shared target critic forward over 2B) ---
    mpo.actor.train()
    mpo.critic.train()
    mpo.target_actor.eval()
    mpo.target_critic.eval()

    state      = torch.zeros((B, *obs_shape), device=device, dtype=torch.float32)
    next_state = torch.zeros((B, *obs_shape), device=device, dtype=torch.float32)
    action     = torch.zeros((B, *act_shape), device=device, dtype=torch.float32)
    reward     = torch.zeros((B,), device=device, dtype=torch.float32)

    gamma = float(getattr(args, "discount_factor", 0.99))

    with torch.no_grad():
        # Build shared (2B) state batch
        all_states = torch.cat([state, next_state], dim=0)  # (2B, obs)

        # Sample actions from target policy for ALL states at once -> (S, 2B, act)
        mu_all, std_all = mpo.target_actor.forward(all_states)        # (2B, act)
        pi_all = Independent(Normal(mu_all, std_all), 1)
        all_sampled_actions = pi_all.sample((S,))                     # (S, 2B, act)

        # Shared target critic forward -> (S, 2B)
        all_states_exp = all_states.unsqueeze(0).expand(S, -1, -1)    # (S, 2B, obs)
        all_q = mpo.target_critic.forward(
            all_states_exp.reshape(S * 2 * B, *obs_shape),
            all_sampled_actions.reshape(S * 2 * B, *act_shape),
        ).view(S, 2 * B)

        # Split into Q(s,a) and Q(s',a')
        next_q = all_q[:, B:]                       # (S, B)
        expected_next_q = next_q.mean(dim=0)        # (B,)

        y = reward + gamma * expected_next_q        # (B,)

    q = mpo.critic.forward(state, action).view(-1)  # (B,)
    critic_loss = F.mse_loss(q, y)
    critic_loss.backward()
    mpo.critic.zero_grad(set_to_none=True)
    print("   - Critic backward warmup done.")

    # --- ZONE 3: ACTOR (M-step-like, using shared (S,2B) sampling) ---
    with torch.no_grad():
        all_states = torch.cat([state, next_state], dim=0)  # (2B, obs)

        mu_all, std_all = mpo.target_actor.forward(all_states)  # (2B, act)
        pi_all = Independent(Normal(mu_all, std_all), 1)
        all_sampled_actions = pi_all.sample((S,))               # (S, 2B, act)

        all_states_exp = all_states.unsqueeze(0).expand(S, -1, -1)  # (S, 2B, obs)
        all_q = mpo.target_critic.forward(
            all_states_exp.reshape(S * 2 * B, *obs_shape),
            all_sampled_actions.reshape(S * 2 * B, *act_shape),
        ).view(S, 2 * B)

        # Weights ONLY for the first B states
        q_sa = all_q[:, :B]  # (S, B)

        # Take eta from mpo if it exists, else use 1.0 (fixed warmup value)
        if hasattr(mpo, "eta_dual"):
            eta_val = float(mpo.eta_dual.detach().cpu().item())
        else:
            eta_val = 1.0
        eta_val = max(eta_val, 1e-6)

        weights = torch.softmax(q_sa / eta_val, dim=0)  # (S, B)

        # For log-prob objective we need actions and (old mean/std) for FIRST B only
        sampled_actions = all_sampled_actions[:, :B, :]  # (S, B, act)
        mu_b  = mu_all[:B, :]                            # (B, act)
        std_b = std_all[:B, :]                           # (B, act)

    mu, std = mpo.actor.forward(state)  # (B, act)

    # Split objective (like your MPO M-step): new mean with old std, and old mean with new std
    pi1 = Independent(Normal(mu,   std_b), 1)
    pi2 = Independent(Normal(mu_b, std),   1)

    logp1 = pi1.log_prob(sampled_actions)  # (S, B)
    logp2 = pi2.log_prob(sampled_actions)  # (S, B)

    actor_loss = -(weights * (logp1 + logp2)).mean()
    actor_loss.backward()

    mpo.actor.zero_grad(set_to_none=True)
    print("   - Actor backward warmup done.")

    print("✅ torch.compile warmup complete.")
    return mpo
