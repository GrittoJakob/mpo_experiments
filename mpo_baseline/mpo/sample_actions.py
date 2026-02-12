import torch


def sample_actions_from_target_actor(self, state_batch, next_state_batch = None, sample_num = 20):
    
    # This function samples actions from the target actor.
    # Inputs:
    #     - state-batch [B, obs_dim]
    #     - next_state_batch [B, obs_dim]
    
    # Function concatenates state batches for shared forward pass in target actor.
    # Returns:
    #     - sampled actions 
    #     - Mu and std from target actor
    
    B = state_batch.size(0)
    with torch.no_grad():

        if next_state_batch is not None:
            all_states = torch.cat([state_batch, next_state_batch], dim=0)  # (2B, obs_dim)

            # get distribution
            all_sampled_actions, mu_off, std_off = self.target_actor.sample_action(all_states, sample_num) # (2B,)
            all_sampled_actions  = all_sampled_actions.permute( 1, 0, 2).contiguous()    #(sample_num, 2B, action_dim)
            sampled_actions      = all_sampled_actions[:, :B]  #(sample_num, B, action_dim)
            return  all_sampled_actions, sampled_actions, mu_off[:B], std_off[:B]
        else:
            sampled_actions, b_mu, b_std = self.target_actor.sample_action(state_batch, sample_num)
            sampled_actions = sampled_actions.permute(1,0,2)
            return sampled_actions, b_mu, b_std



