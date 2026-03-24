import torch
from .plot_dist import plot_mpo_weight_distribution

    
def expectation_step_eval(mpo, target_q, sampled_actions, collect_stats: bool=False):
    """
    E-step of MPO:
    - Evaluate Q-values with the target critic.
    - Update the temperature parameter eta via the dual function.
    - Compute weight action penalties via the temperature loss function
    - Compute normalized weights over actions (norm_target_q).
    """
    diff_out_of_bound = None

    # Dual function returns a scalar that we differentiate w.r.t. eta and our normalized values
    norm_target_q, loss_temperature = compute_weights_temperature_loss(mpo.eta_dual, target_q, mpo.eps_dual)
    loss_dual = loss_temperature

     # Plot: Original E-step
    plot_mpo_weight_distribution(
        sampled_actions=sampled_actions,
        weights=norm_target_q,
        b_idx=0,  # batch_size=1
        title="Original MPO E-step weights",
        action_low=mpo.action_space_low,
        action_high=mpo.action_space_high,
    )

    # Initialize action penalty variables with None
    has_oob = None
    penalty_normalized_weights = None
    loss_penalty_temperature = None
    diff_out_of_bound = None
    
    # Action Penalty
    # Compute quadratic curve when out of bound:
    actions_squashed  = torch.max(torch.min(sampled_actions, mpo.action_space_high), self.action_space_low)
    diff_out_of_bound = sampled_actions - actions_squashed
    has_oob = (diff_out_of_bound.abs().amax() > 0).to(norm_target_q.dtype)
    
    cost_out_of_bound = -(diff_out_of_bound.pow(2).sum(dim=-1))  # (N,B)

    # Compute dual function for eta and get weights
    penalty_normalized_weights, loss_penalty_temperature = compute_weights_temperature_loss(
        mpo.eta_penalty, cost_out_of_bound, mpo.eps_penalty
        )
    
    # Plot: Action penalty weights
    plot_mpo_weight_distribution(
        sampled_actions=sampled_actions,
        weights=penalty_normalized_weights,
        b_idx=0,
        title="Action penalty weights",
        action_low=mpo.action_space_low,
        action_high=mpo.action_space_high,
    )


    # Mix both weight distributions of original E-Step and action penalty by a mixing factor lambda
    lam_eff = mpo.lam * has_oob
    norm_target_q = (1 - lam_eff) * norm_target_q + lam_eff * penalty_normalized_weights
    norm_target_q = norm_target_q / (norm_target_q.sum(dim=0, keepdim=True) + 1e-8)
    loss_dual = (1 - lam_eff) * loss_dual + lam_eff * loss_penalty_temperature
    
    # Plot: Final distribution
    plot_mpo_weight_distribution(
        sampled_actions=sampled_actions,
        weights=norm_target_q,
        b_idx=0,
        title="Final MPO E-step distribution",
        action_low=mpo.action_space_low,
        action_high=mpo.action_space_high,
    )

    return norm_target_q


def compute_weights_temperature_loss(temperature: torch.Tensor, values: torch.Tensor, epsilon: float):
    """
    temperature: torch scalar, requires_grad=True
    values: (N, B)  (No Gradients)
    epsilon: float
    returns:
    weights: (N, B) detached
    dual_loss: scalar tensor (has Grad wrt temperature)
    """
    values = values.detach()  # stop grad for 

    # weights (stop grad wrt temperature) 
    tempered = values / temperature.detach()
    tempered = tempered - tempered.max(dim=0, keepdim=True).values  # stable
    weights = torch.softmax(tempered, dim=0).detach()

    # dual loss (Grad wrt temperature) 
    values_T = values.transpose(0, 1)  # (B, N)
    max_v = values_T.max(dim=1, keepdim=True).values  # (B, 1)
    exp_term = torch.exp((values_T - max_v) / temperature)  # grad fließt in temperature
    log_mean_exp = torch.log(exp_term.mean(dim=1) + 1e-8)  # (B,)

    dual_loss = temperature * epsilon + max_v.mean() + temperature * log_mean_exp.mean()


    return weights, dual_loss