import torch
import matplotlib.pyplot as plt
from .helpers.plot_e_step import plot_mpo_weight_distribution
from mpo.algorithm.expectation_step import compute_weights_temperature_loss
import wandb

 
def expectation_step_eval(mpo, target_q,mu_off, std_off, sampled_actions, writer=None, step=None):
    
    # Original E-Step: Compute weights by dual function
    original_target_q, loss_temperature = compute_weights_temperature_loss(
        mpo.eta_dual, target_q, mpo.eps_dual
    )
    loss_dual = loss_temperature

    # Visualize and log in wandb
    fig = plot_mpo_weight_distribution(
        sampled_actions=sampled_actions,
        weights=original_target_q,
        mu_dist = mu_off,
        std_dist = std_off,
        b_idx=0,
        title="Original MPO E-step weights",
        action_low=mpo.action_space_low,
        action_high=mpo.action_space_high,
    )

    log_figure(writer, fig, "plots_dist/original_weights", step)

    # Action Penalty from MO-MPO:
    action_low = torch.as_tensor(
        mpo.action_space_low, device=sampled_actions.device, dtype=sampled_actions.dtype
    )
    action_high = torch.as_tensor(
        mpo.action_space_high, device=sampled_actions.device, dtype=sampled_actions.dtype
    )

    actions_squashed = torch.max(torch.min(sampled_actions, action_high), action_low)
    diff_out_of_bound = sampled_actions - actions_squashed
    has_oob = (diff_out_of_bound.abs().amax() > 0).to(original_target_q.dtype)

    cost_out_of_bound = -(diff_out_of_bound.pow(2).sum(dim=-1))  # (N, B)

    penalty_normalized_weights, loss_penalty_temperature = compute_weights_temperature_loss(
        mpo.eta_penalty, cost_out_of_bound, mpo.eps_penalty
    )

    fig = plot_mpo_weight_distribution(
        sampled_actions=sampled_actions,
        weights=penalty_normalized_weights,
        mu_dist = mu_off,
        std_dist= std_off,
        b_idx=0,
        title="Action penalty weights",
        action_low=mpo.action_space_low,
        action_high=mpo.action_space_high,
    )
    log_figure(writer, fig, "plots_dist/action_penalty_weights", step)

    lam_eff = mpo.lam * has_oob
    norm_target_q = (1 - lam_eff) * original_target_q + lam_eff * penalty_normalized_weights
    final_q_dist = norm_target_q / (norm_target_q.sum(dim=0, keepdim=True) + 1e-8)
    loss_dual = (1 - lam_eff) * loss_dual + lam_eff * loss_penalty_temperature

    fig = plot_mpo_weight_distribution(
        sampled_actions=sampled_actions,
        weights=norm_target_q,
        mu_dist= mu_off,
        std_dist = std_off,
        b_idx=0,
        title="Final MPO E-step distribution",
        action_low=mpo.action_space_low,
        action_high=mpo.action_space_high,
    )
    log_figure(writer, fig, "plots_dist/final_distribution", step)

    return final_q_dist, original_target_q



    

def log_figure(writer, fig, tag: str, step: int):

    writer.add_figure(tag, fig, global_step=step)
    wandb.log({tag: wandb.Image(fig)}, step=step)
    plt.close(fig)