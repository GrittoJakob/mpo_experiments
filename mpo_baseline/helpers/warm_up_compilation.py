import torch
import gymnasium as gym

def compile_mpo_modules(mpo, *, mode="reduce-overhead", fullgraph=False, dynamic=False):
    if not hasattr(torch, "compile"):
        print("⚠️ torch.compile not available. Skipping.")
        return mpo

    mpo.actor.forward         = torch.compile(mpo.actor.forward,         mode=mode, fullgraph=fullgraph, dynamic=dynamic)
    mpo.critic.forward        = torch.compile(mpo.critic.forward,        mode=mode, fullgraph=fullgraph, dynamic=dynamic)
    mpo.target_actor.forward  = torch.compile(mpo.target_actor.forward,  mode=mode, fullgraph=fullgraph, dynamic=dynamic)
    mpo.target_critic.forward = torch.compile(mpo.target_critic.forward, mode=mode, fullgraph=fullgraph, dynamic=dynamic)

    # WICHTIG: NICHT mpo.sample_actions_from_target_actor / target_critic_forward_pass compilieren,
    # solange dort Views zurückgegeben werden.
    return mpo


def warmup_mpo_compile(args, device, env, mpo, *, compile_mode="reduce-overhead"):
    print(f"🔥 torch.compile warmup on {device} ...")

    obs_dim = int(mpo.state_dim)
    act_dim = int(mpo.action_dim)
    B = int(args.batch_size)
    N = int(mpo.sample_action_num)  # = args.sample_action_num

    mpo = compile_mpo_modules(mpo, mode=compile_mode, fullgraph=False, dynamic=False)

    def mark_step():
        if hasattr(torch, "compiler") and hasattr(torch.compiler, "cudagraph_mark_step_begin"):
            torch.compiler.cudagraph_mark_step_begin()

    # ---- ZONE 1: Rollout Shapes (AsyncVectorEnv = VectorEnv) ----
    rollout_bs = {1, B}
    if isinstance(env, gym.vector.VectorEnv):
        rollout_bs.add(int(env.num_envs))
    rollout_bs = sorted(rollout_bs)

    mpo.actor.eval()
    mpo.target_actor.eval()
    with torch.no_grad():
        for bs in rollout_bs:
            mark_step()
            obs = torch.zeros((bs, obs_dim), device=device, dtype=torch.float32)
            _ = mpo.actor.forward(obs)
            _ = mpo.target_actor.forward(obs)

    print(f"   - Rollout warmup done (bs={rollout_bs}).")

    # ---- ZONE 2: Update path (Shapes wie train_loop) ----
    mpo.actor.train()
    mpo.critic.train()
    mpo.target_actor.eval()
    mpo.target_critic.eval()

    state_batch      = torch.zeros((B, obs_dim), device=device, dtype=torch.float32)
    next_state_batch = torch.zeros((B, obs_dim), device=device, dtype=torch.float32)
    action_batch     = torch.zeros((B, act_dim), device=device, dtype=torch.float32)
    reward_batch     = torch.zeros((B,),         device=device, dtype=torch.float32)
    terminated_batch = torch.zeros((B,),         device=device, dtype=torch.float32)
    truncated_batch  = torch.zeros((B,),         device=device, dtype=torch.float32)

    # sample_actions_from_target_actor (eager wrapper, compiled target_actor.forward dahinter)
    mark_step()
    all_sampled_actions, sampled_actions, mu_off, std_off = mpo.sample_actions_from_target_actor(
        state_batch=state_batch,
        next_state_batch=next_state_batch,
        sample_num=N,
    )
    # Wenn du (noch) nicht den contiguous-Patch drin hast:
    # sampled_actions = sampled_actions.contiguous()
    # mu_off = mu_off.contiguous()
    # std_off = std_off.contiguous()

    print("   - sample_actions_from_target_actor warmup done.")

    # target_critic_forward_pass (eager wrapper, compiled target_critic.forward dahinter)
    mark_step()
    target_q, next_target_q = mpo.target_critic_forward_pass(
        state_batch=state_batch,
        next_state_batch=next_state_batch,
        all_sampled_actions=all_sampled_actions,
    )
    # wenn du den contiguous-Patch noch nicht drin hast:
    # target_q = target_q.contiguous()
    # next_target_q = next_target_q.contiguous()

    print("   - target_critic_forward_pass warmup done.")

    # critic_update_td
    mark_step()
    _ = mpo.critic_update_td(
        next_target_q=next_target_q,
        state_batch=state_batch,
        action_batch=action_batch,
        reward_batch=reward_batch,
        terminated_batch=terminated_batch,
        truncated_batch=truncated_batch,
        collect_stats=False,
    )
    print("   - critic_update_td warmup done.")

    # expectation_step
    mark_step()
    norm_target_q, _ = mpo.expectation_step(
        target_q=target_q,
        sampled_actions=sampled_actions,
        collect_stats=False,
    )
    print("   - expectation_step warmup done.")

    # maximization_step
    mark_step()
    _ = mpo.maximization_step(
        state_batch=state_batch,
        norm_target_q=norm_target_q,
        sampled_actions=sampled_actions,
        mu_off=mu_off,
        std_off=std_off,
        collect_stats=False,
    )
    print("   - maximization_step warmup done.")

    print("✅ torch.compile warmup complete.")
    return mpo
