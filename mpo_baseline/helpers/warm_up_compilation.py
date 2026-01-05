import torch
import torch.nn.functional as F
from torch.distributions import Independent, Normal


def _space_shape(env, which: str):
    """
    Works for both gym.Env and gym.vector.VectorEnv:
    - VectorEnv has single_observation_space / single_action_space
    - Env has observation_space / action_space
    """
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
    """
    Compiles ONLY the hot forward paths.
    Important: in your code, actor.action()/evaluate_action() call self.forward(...)
    directly (not self(...)). So we compile + replace .forward explicitly.
    """
    if not hasattr(torch, "compile"):
        print("⚠️ torch.compile not available in this torch version. Skipping.")
        return mpo

    # Compile forward methods (best bang-for-buck + actually used inside action/evaluate_action)
    mpo.actor.forward = torch.compile(mpo.actor.forward, mode=mode, fullgraph=fullgraph, dynamic=dynamic)
    mpo.critic.forward = torch.compile(mpo.critic.forward, mode=mode, fullgraph=fullgraph, dynamic=dynamic)

    mpo.target_actor.forward = torch.compile(mpo.target_actor.forward, mode=mode, fullgraph=fullgraph, dynamic=dynamic)
    mpo.target_critic.forward = torch.compile(mpo.target_critic.forward, mode=mode, fullgraph=fullgraph, dynamic=dynamic)

    return mpo


@torch.no_grad()
def _rollout_kernel(actor, obs):
    # forward used by rollout (and indirectly by actor.action())
    mu, std = actor.forward(obs)
    dist = Independent(Normal(mu, std), 1)
    a = dist.sample()
    return a


def warmup_mpo_compile(args, device, env, mpo, *, compile_mode: str = "reduce-overhead"):
    """
    Warmup for torch.compile (non-recurrent version).
    Runs representative forward/backward passes so compilation happens
    BEFORE your real training loop starts.
    """
    print(f"🔥 torch.compile warmup on {device} ...")

    obs_shape = _space_shape(env, "obs")
    act_shape = _space_shape(env, "act")

    B = int(args.batch_size)
    S = int(args.sample_action_num)

    # 0) (Optional) compile hot paths
    mpo = compile_mpo_modules(mpo, mode=compile_mode, fullgraph=False, dynamic=False)

    # --- ZONE 1: ROLLOUT SHAPES (no grad) ---
    # Rollout is typically batch=1; training is batch=B.
    for bs in (1,1):
        obs = torch.zeros((bs, *obs_shape), device=device, dtype=torch.float32)
        _ = _rollout_kernel(mpo.actor, obs)
        _ = _rollout_kernel(mpo.target_actor, obs)
    print("   - Rollout forward warmup done.")

    # --- ZONE 2: CRITIC TRAINING (forward + backward) ---
    # Mimic critic_update_td core compute (but WITHOUT optimizer.step()).
    mpo.actor.train()
    mpo.critic.train()
    mpo.target_actor.eval()
    mpo.target_critic.eval()

    state = torch.zeros((B, *obs_shape), device=device, dtype=torch.float32)
    action = torch.zeros((B, *act_shape), device=device, dtype=torch.float32)
    next_state = torch.zeros((B, *obs_shape), device=device, dtype=torch.float32)
    reward = torch.zeros((B,), device=device, dtype=torch.float32)

    # target y = r + gamma * E_a'[Q_targ(s',a')]
    with torch.no_grad():
        # sample S actions for next_state from target policy
        mu_ns, std_ns = mpo.target_actor.forward(next_state)  # actor returns mean,std :contentReference[oaicite:2]{index=2}
        pi_ns = Independent(Normal(mu_ns, std_ns), 1)
        a_ns = pi_ns.sample((S,))  # (S,B,act)
        s_ns = next_state.unsqueeze(0).expand(S, -1, -1)      # (S,B,obs)

        q_ns = mpo.target_critic.forward(
            s_ns.reshape(S * B, *obs_shape),
            a_ns.reshape(S * B, *act_shape),
        ).reshape(S, B)

        expected_next_q = q_ns.mean(dim=0)  # (B,)
        gamma = float(getattr(args, "discount_factor", 0.99))
        y = reward + gamma * expected_next_q

    q = mpo.critic.forward(state, action).squeeze(-1)  # (B,)
    critic_loss = F.mse_loss(q, y)
    critic_loss.backward()

    mpo.critic.zero_grad(set_to_none=True)
    print("   - Critic backward warmup done.")

    # --- ZONE 3: ACTOR TRAINING (M-step-like loss + backward) ---
    # Build (sampled_actions, weights) similarly to E-step, then do weighted log-prob loss.
    with torch.no_grad():
        mu_b, std_b = mpo.target_actor.forward(state)
        
        if device.type == "cuda":
            mu_b = mu_b.clone()
            std_b = std_b.clone()
        pi_b = Independent(Normal(mu_b, std_b), 1)
        sampled_actions = pi_b.sample((S,))  # (S,B,act)

        s_exp = state.unsqueeze(0).expand(S, -1, -1)
        q_sa = mpo.target_critic.forward(
            s_exp.reshape(S * B, *obs_shape),
            sampled_actions.reshape(S * B, *act_shape),
        ).reshape(S, B)

        # weights ~ softmax(Q / eta) (eta fixed just for warmup)
        eta = float(getattr(mpo, "eta", 1.0)) if hasattr(mpo, "eta") else 1.0
        weights = torch.softmax(q_sa / max(eta, 1e-6), dim=0)  # (S,B)

    # current policy params
    mu, std = mpo.actor.forward(state)
    if device.type == "cuda":
        mu = mu.clone()
        std = std.clone()

    # "paper2 style" split objective like in your mpo.maximization_step :contentReference[oaicite:3]{index=3}
    pi1 = Independent(Normal(mu, std_b), 1)   # new mean, old std
    pi2 = Independent(Normal(mu_b, std), 1)   # old mean, new std

    logp1 = pi1.log_prob(sampled_actions)     # (S,B)
    logp2 = pi2.log_prob(sampled_actions)     # (S,B)

    actor_loss = -(weights * (logp1 + logp2)).mean()
    actor_loss.backward()

    mpo.actor.zero_grad(set_to_none=True)
    print("   - Actor backward warmup done.")

    print("✅ torch.compile warmup complete.")
    return mpo
