import torch


def sample_actions_from_target_actor(self, state_batch, next_state_batch = None, sample_num = 20):
        B = state_batch.size(0)
        with torch.no_grad():

            if next_state_batch is not None:
                all_states = torch.cat([state_batch, next_state_batch], dim=0)  # (2B, obs_dim)

                # get distribution
                all_sampled_actions, b_mu, b_std = self.target_actor.sample_action(all_states, sample_num) # (2B,)
                all_sampled_actions  = all_sampled_actions.permute( 1, 0, 2).contiguous()    #(sample_num, 2B, action_dim)
                sampled_actions      = all_sampled_actions[:, :B]  #(sample_num, B, action_dim)
                sampled_next_actions = all_sampled_actions[:, B:]  #(sample_num, B, action_dim)
                return  all_sampled_actions, sampled_actions, b_mu[:B], b_std[:B]
            else:
                sampled_actions, b_mu, b_std = self.target_actor.sample_action(state_batch, sample_num)
                sampled_actions = sampled_actions.permute(1,0,2)
                return sampled_actions, b_mu, b_std



    
# def assert_sampled_actions_shape(self, sampled_actions, state_batch, name="sampled_actions"):
#     """
#     Ensures sampled_actions has shape (N, B, action_dim)
#     where:
#     sample_num = number of sampled actions per state
#     B = state_batch.size(0)
#     action_dim = self.action_dim
#     """

#     assert sampled_actions is not None, f"{name} is None"
#     assert sampled_actions.dim() == 3, \
#         f"{name} must be 3D (sample_num, B, act_dim), got shape {tuple(sampled_actions.shape)}"

#     N, B, A = sampled_actions.shape
#     expected_B = state_batch.size(0)

#     assert N == self.sample_action_num, \
#         f"{name}: Expected N={self.sample_action_num} actions per state, got N={N}"
#     assert B == expected_B, \
#         f"{name}: Expected B={expected_B} states, got K={B}"
#     assert A == self.action_dim, \
#         f"{name}: Expected action_dim={self.action_dim}, got A={A}"
