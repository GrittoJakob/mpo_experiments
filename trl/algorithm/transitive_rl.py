import torch
import math
import torch.nn.functional as F



def trl_critic_loss(self, batch, collect_stats = None):
    
    # Get data from batch
    
    # Obs
    start_obs = batch["start_obs"]
    midpoint_obs = batch["midpoint_obs"]

    # Goals
    goal = batch["goal"]
    midpoint_goal = batch["midpoint_goal"]

    # Relabel obs
    midpoint_to_goal = self.obs_wrapper.replace_desired_goal(midpoint_obs.clone(), goal)
    start_to_midpoint = self.obs_wrapper.replace_desired_goal(start_obs.clone(), midpoint_goal)
    start_to_goal = self.obs_wrapper.replace_desired_goal(start_obs.clone(), goal)

    # Actions
    start_actions = batch["actions"]
    midpoint_actions = batch["midpoint_actions"]

    # Offsets
    goal_offset = batch["offset"].unsqueeze(-1).float()
    first_offset = batch["midpoint_offset"].unsqueeze(-1).float()
    second_offset = goal_offset - first_offset

    # Online Critic
    q_logits = self.critic(start_to_goal, start_actions)     # [B, 1] or [B]
    if q_logits.dim() == 1:
        q_logits = q_logits.unsqueeze(-1)

    critic = torch.sigmoid(q_logits)

    with torch.no_grad():
        # Target Critic
        first_q_logits = self.target_critic(start_to_midpoint, start_actions)
        second_q_logits = self.target_critic(midpoint_to_goal, midpoint_actions)

        if first_q_logits.dim() == 1:
            first_q_logits = first_q_logits.unsqueeze(-1)
        if second_q_logits.dim() == 1:
            second_q_logits = second_q_logits.unsqueeze(-1)

        # Case 1:
        # if k - i <= 1, then compute gamma^(k-i)
        first_q = torch.where(
            first_offset <= 1,
            self.gamma ** first_offset,
            torch.sigmoid(first_q_logits),
        )

        # Case 2:
        # If j - k <= 1, then gamma^(j-k)
        second_q = torch.where(
            second_offset <= 1,
            self.gamma ** second_offset,
            torch.sigmoid(second_q_logits),
        )

        # Transitive Target
        target_critic = first_q * second_q
        target_critic = target_critic.clamp(min=1e-8, max=1.0)

        # Expectile Weight
        expectile_weight = torch.where(
            target_critic >= critic,
            torch.full_like(target_critic, self.expectile),
            torch.full_like(target_critic, 1.0 - self.expectile),
        )

        # Distance-based weight
        dist = torch.log(target_critic.clamp(min=1e-8)) / math.log(self.gamma)
        dist_weight = (1.0 / (1.0 + dist)) ** self.lamda_trl
   

    self.critic_optimizer.zero_grad(set_to_none=True)
       
    critic_loss = expectile_weight * dist_weight * F.binary_cross_entropy(
        critic, target_critic, reduction="none"
    )
    loss = critic_loss.mean()
    loss.backward()
    self.critic_optimizer.step()

    if collect_stats:
        stats = {
            "critic_loss": loss.detach().mean(),
            "q_current_mean": critic.detach().mean(),
            "q_target_mean": target_critic.detach().mean(),
        }
    else:
        stats = None
    return stats

