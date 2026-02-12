import torch

def critic_update_td(self, next_target_q,  state_batch, action_batch, reward_batch, terminated_batch, truncated_batch, collect_stats):
    
    # TD Learning algorithm:
    # Delta_Q = Q_t - (reward + gamma * Q_t+1)
    
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
