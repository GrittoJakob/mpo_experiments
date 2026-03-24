import torch
from mpo.algorithm.__init__ import MPO
from buffer.replaybuffer import ReplayBuffer


def script_e_step_eval(
    args,
    mpo,
    replaybuffer,
    ):

    # Sample mini batch from buffer
    batch = replaybuffer.sample_batch(1)
    obs_batch = batch["obs"]
    next_obs_batch = batch["next_obs"]
    actions_batch = batch["actions"]
    rewards_batch = batch["rewards"]
    truncated_batch = batch["truncated"]
    terminated_batch = batch["terminated"]

    collect_stats = False

    # Compute target_actor forward pass to get sampled_actions an b_mu, b_st
    # all_sampled_actions: sampled actions for timestep t and t+1 concatenated
    # sampled actions: sampled actions only for timestep t
    all_sampled_actions, sampled_actions, mu_off, std_off = mpo.sample_actions_from_target_actor(
        state_batch= obs_batch,
        next_state_batch= next_obs_batch,
        sample_num= args.sample_action_num
    )

    # Compute forward pass of target critic for state_batch and next_state_batch
    target_q, next_target_q = mpo.shared_target_critic_forward_pass(
        state_batch = obs_batch,
        next_state_batch= next_obs_batch,
        all_sampled_actions = all_sampled_actions
    )

    # Policy evaluation (critic update)
    __ = mpo.td_learning( 
        next_target_q = next_target_q,
        state_batch = obs_batch, 
        action_batch =actions_batch, 
        reward_batch= rewards_batch, 
        terminated_batch = terminated_batch,
        truncated_batch= truncated_batch,
        collect_stats= collect_stats
        )
    
    _, _ = mpo.(
        target_q= target_q,
        sampled_actions=sampled_actions, 
        )