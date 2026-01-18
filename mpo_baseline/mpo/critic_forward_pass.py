import torch


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