import copy
import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from torch.distributions import Independent, Normal
from torch.nn.utils import clip_grad_norm_

from helpers.Gaussian_KL_div import gaussian_kl_diag



def maximization_step_eval(
    mpo,
    num_iterations,
    state_batch,
    norm_target_q,
    sampled_actions,
    mu_off,
    std_off,
    writer=None,
    step=None,
    b_idx: int = 0,
):
    """
    Run an evaluation-only M-step rollout on a fixed batch.

    The function visualizes:
    - the target distribution `norm_target_q`,
    - the online actor distribution before the first gradient update,
    - the online actor distribution after every gradient update,
    - and the ELBO-style objective evolution across iterations.

    Important:
    This function restores the original actor parameters, optimizer state,
    and Lagrange multipliers after the evaluation plot is created.
    Therefore, it does not alter training.
    """
    _validate_m_step_eval_inputs(state_batch, norm_target_q, sampled_actions, mu_off, std_off, b_idx)

    actor_was_training = mpo.actor.training
    actor_state_backup = copy.deepcopy(mpo.actor.state_dict())
    optimizer_state_backup = copy.deepcopy(mpo.actor_optimizer.state_dict())
    eta_mu_backup = mpo.eta_mu.detach().clone()
    eta_sigma_backup = mpo.eta_sigma.detach().clone()

    snapshots = []

    try:
        mpo.actor.train()

        # Snapshot before the first gradient update.
        snapshots.append(
            _collect_m_step_snapshot(
                mpo=mpo,
                state_batch=state_batch,
                norm_target_q=norm_target_q,
                sampled_actions=sampled_actions,
                mu_off=mu_off,
                std_off=std_off,
                iteration=0,
                label="before update",
            )
        )

        for iteration in range(1, num_iterations + 1):
            _run_single_m_step_update(
                mpo=mpo,
                state_batch=state_batch,
                norm_target_q=norm_target_q,
                sampled_actions=sampled_actions,
                mu_off=mu_off,
                std_off=std_off,
            )

            snapshots.append(
                _collect_m_step_snapshot(
                    mpo=mpo,
                    state_batch=state_batch,
                    norm_target_q=norm_target_q,
                    sampled_actions=sampled_actions,
                    mu_off=mu_off,
                    std_off=std_off,
                    iteration=iteration,
                    label=f"after update {iteration}",
                )
            )

        fig = plot_m_step_distribution_evolution(
            sampled_actions=sampled_actions,
            norm_target_q=norm_target_q,
            snapshots=snapshots,
            b_idx=b_idx,
            title="M-step distribution evolution",
            action_low=getattr(mpo, "action_space_low", None),
            action_high=getattr(mpo, "action_space_high", None),
        )
        log_figure(writer, fig, "plots_dist/distribution_evolution", step)

        return {
            "iterations": [snap["iteration"] for snap in snapshots],
            "policy_objective": [snap["policy_objective"] for snap in snapshots],
            "elbo": [snap["elbo"] for snap in snapshots],
            "constraint_penalty": [snap["constraint_penalty"] for snap in snapshots],
            "mean_kl_mu": [snap["mean_kl_mu"] for snap in snapshots],
            "mean_kl_sigma": [snap["mean_kl_sigma"] for snap in snapshots],
        }

    finally:
        mpo.actor.load_state_dict(actor_state_backup)
        mpo.actor_optimizer.load_state_dict(optimizer_state_backup)
        mpo.eta_mu.data.copy_(eta_mu_backup)
        mpo.eta_sigma.data.copy_(eta_sigma_backup)

        if actor_was_training:
            mpo.actor.train()
        else:
            mpo.actor.eval()



def _run_single_m_step_update(mpo, state_batch, norm_target_q, sampled_actions, mu_off, std_off):
    """Run exactly one gradient update of the MPO M-step objective."""
    mu_on, std_on = mpo.actor.forward(state_batch)

    # Decoupled policy distributions used in MPO.
    pi_1 = Independent(Normal(mu_on, std_off), 1)
    pi_2 = Independent(Normal(mu_off, std_on), 1)
    logp1 = pi_1.log_prob(sampled_actions)
    logp2 = pi_2.log_prob(sampled_actions)

    weighted_logp = norm_target_q * (logp1 + logp2)
    batch_size = state_batch.shape[0]
    loss_p = -(weighted_logp).sum() / batch_size

    C_mu_dim, C_sigma_dim, _, _ = gaussian_kl_diag(
        mu_off,
        mu_on,
        std_off,
        std_on,
        mpo.use_mass_force_KL,
    )

    with torch.no_grad():
        mpo.eta_mu -= mpo.alpha_mu_scale * (mpo.eps_mu_dim - C_mu_dim)
        mpo.eta_sigma -= mpo.alpha_sigma_scale * (mpo.eps_gamma_dim - C_sigma_dim)
        mpo.eta_mu.clamp_(1e-10, mpo.alpha_mu_max)
        mpo.eta_sigma.clamp_(1e-10, mpo.alpha_sigma_max)

    constraint_mu = C_mu_dim - mpo.eps_mu_dim
    constraint_sigma = C_sigma_dim - mpo.eps_gamma_dim

    loss_l = (
        loss_p
        + (mpo.eta_mu * constraint_mu).sum()
        + (mpo.eta_sigma * constraint_sigma).sum()
    )

    mpo.actor_optimizer.zero_grad(set_to_none=True)
    loss_l.backward()
    clip_grad_norm_(mpo.actor.parameters(), 0.1)
    mpo.actor_optimizer.step()



def _collect_m_step_snapshot(
    mpo,
    state_batch,
    norm_target_q,
    sampled_actions,
    mu_off,
    std_off,
    iteration: int,
    label: str,
):
    """
    Collect actor statistics for one point in the M-step update trajectory.

    We track an ELBO-style objective as `-loss_l`, because `loss_l` is the
    quantity that is minimized during the actor update.
    """
    with torch.no_grad():
        mu_on, std_on = mpo.actor.forward(state_batch)

        pi_1 = Independent(Normal(mu_on, std_off), 1)
        pi_2 = Independent(Normal(mu_off, std_on), 1)
        logp1 = pi_1.log_prob(sampled_actions)
        logp2 = pi_2.log_prob(sampled_actions)

        weighted_logp = norm_target_q * (logp1 + logp2)
        batch_size = state_batch.shape[0]
        policy_objective = weighted_logp.sum() / batch_size

        C_mu_dim, C_sigma_dim, _, _ = gaussian_kl_diag(
            mu_off,
            mu_on,
            std_off,
            std_on,
            mpo.use_mass_force_KL,
        )

        constraint_mu = C_mu_dim - mpo.eps_mu_dim
        constraint_sigma = C_sigma_dim - mpo.eps_gamma_dim
        constraint_penalty = (
            (mpo.eta_mu * constraint_mu).sum()
            + (mpo.eta_sigma * constraint_sigma).sum()
        )

        elbo = policy_objective - constraint_penalty

        return {
            "iteration": int(iteration),
            "label": label,
            "mu_on": mu_on.detach().cpu().clone(),
            "std_on": std_on.detach().cpu().clone(),
            "policy_objective": float(policy_objective.item()),
            "constraint_penalty": float(constraint_penalty.item()),
            "elbo": float(elbo.item()),
            "mean_kl_mu": float(C_mu_dim.mean().item()),
            "mean_kl_sigma": float(C_sigma_dim.mean().item()),
        }



def plot_m_step_distribution_evolution(
    sampled_actions: torch.Tensor,
    norm_target_q: torch.Tensor,
    snapshots,
    b_idx: int = 0,
    title: str = "",
    action_low=None,
    action_high=None,
    show: bool = False,
):
    """
    Create one figure with two subplots:
    1. First action dimension: target distribution and actor evolution.
    2. Last action dimension: target distribution and actor evolution.
    """
    actions_np = sampled_actions.detach().cpu().numpy()
    target_np = norm_target_q.detach().cpu().numpy()

    x_first = actions_np[:, b_idx, 0]
    x_last = actions_np[:, b_idx, -1]
    target_weights = target_np[:, b_idx]

    action_dim = sampled_actions.shape[-1]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(title if title else f"M-step evaluation for batch sample {b_idx}")

    _plot_single_dimension_evolution(
        ax=axes[0],
        x_samples=x_first,
        target_weights=target_weights,
        snapshots=snapshots,
        b_idx=b_idx,
        dim_idx=0,
        dim_name="First actor dimension",
        action_low=action_low,
        action_high=action_high,
    )

    _plot_single_dimension_evolution(
        ax=axes[1],
        x_samples=x_last,
        target_weights=target_weights,
        snapshots=snapshots,
        b_idx=b_idx,
        dim_idx=action_dim - 1,
        dim_name="Last actor dimension",
        action_low=action_low,
        action_high=action_high,
    )

    plt.tight_layout()

    if show:
        plt.show()

    return fig



def _plot_single_dimension_evolution(
    ax,
    x_samples: np.ndarray,
    target_weights: np.ndarray,
    snapshots,
    b_idx: int,
    dim_idx: int,
    dim_name: str,
    action_low=None,
    action_high=None,
):
    order = np.argsort(x_samples)
    x_plot = x_samples[order]
    y_plot = target_weights[order]

    x_min = float(np.min(x_plot))
    x_max = float(np.max(x_plot))

    for snapshot in snapshots:
        mu_val = float(snapshot["mu_on"][b_idx, dim_idx].item())
        std_val = max(float(snapshot["std_on"][b_idx, dim_idx].item()), 1e-6)
        x_min = min(x_min, mu_val - 4.0 * std_val)
        x_max = max(x_max, mu_val + 4.0 * std_val)

    low_val = None
    high_val = None

    if action_low is not None:
        low_val = action_low[dim_idx].detach().cpu().item() if torch.is_tensor(action_low) else action_low[dim_idx]
        x_min = min(x_min, float(low_val))

    if action_high is not None:
        high_val = action_high[dim_idx].detach().cpu().item() if torch.is_tensor(action_high) else action_high[dim_idx]
        x_max = max(x_max, float(high_val))

    x_dense = np.linspace(x_min, x_max, 500)

    # Plot the target distribution once.
    ax.scatter(x_plot, y_plot, s=22, alpha=0.85, label="Target distribution")
    ax.plot(x_plot, y_plot, alpha=0.65)

    # Overlay the online actor distribution before and after every update.
    cmap = plt.cm.viridis(np.linspace(0.15, 0.95, len(snapshots)))
    target_peak = max(float(np.max(y_plot)), 1e-8)

    for color, snapshot in zip(cmap, snapshots):
        mu_val = float(snapshot["mu_on"][b_idx, dim_idx].item())
        std_val = max(float(snapshot["std_on"][b_idx, dim_idx].item()), 1e-6)
        pdf_dense = _gaussian_pdf(x_dense, mu_val, std_val)

        # Rescale the Gaussian peak to the target peak for easier visual comparison.
        pdf_dense = pdf_dense / max(float(np.max(pdf_dense)), 1e-8) * target_peak

        ax.plot(
            x_dense,
            pdf_dense,
            color=color,
            linewidth=2.0,
            alpha=0.95,
            label=f"iter {snapshot['iteration']}: mu={mu_val:.2f}, std={std_val:.2f}",
        )

    if low_val is not None:
        ax.axvline(low_val, linestyle="--", alpha=0.7, label="action_low")

    if high_val is not None:
        ax.axvline(high_val, linestyle="--", alpha=0.7, label="action_high")

    ax.set_xlabel(f"Action value ({dim_name})")
    ax.set_ylabel("Target weight / scaled actor density")
    ax.set_title(dim_name)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, loc="best")



def _gaussian_pdf(x: np.ndarray, mu: float, std: float) -> np.ndarray:
    return (1.0 / (std * np.sqrt(2.0 * np.pi))) * np.exp(-0.5 * ((x - mu) / std) ** 2)



def _validate_m_step_eval_inputs(state_batch, norm_target_q, sampled_actions, mu_off, std_off, b_idx: int):
    if sampled_actions.ndim != 3:
        raise ValueError(f"Expected sampled_actions with shape (N, B, A), got {sampled_actions.shape}")
    if norm_target_q.ndim != 2:
        raise ValueError(f"Expected norm_target_q with shape (N, B), got {norm_target_q.shape}")
    if sampled_actions.shape[:2] != norm_target_q.shape:
        raise ValueError(
            f"sampled_actions.shape[:2]={sampled_actions.shape[:2]} must match norm_target_q.shape={norm_target_q.shape}"
        )
    if state_batch.shape[0] != sampled_actions.shape[1]:
        raise ValueError(
            f"state_batch batch size {state_batch.shape[0]} must match sampled_actions batch size {sampled_actions.shape[1]}"
        )
    if not 0 <= b_idx < sampled_actions.shape[1]:
        raise ValueError(f"b_idx={b_idx} out of range for batch size {sampled_actions.shape[1]}")
    if mu_off.shape != std_off.shape:
        raise ValueError(f"mu_off.shape={mu_off.shape} must match std_off.shape={std_off.shape}")



def log_figure(writer, fig, tag: str, step: int):
    """Log a matplotlib figure to TensorBoard and Weights & Biases."""
    if writer is not None and step is not None:
        writer.add_figure(tag, fig, global_step=step)

    if wandb.run is not None:
        if step is None:
            wandb.log({tag: wandb.Image(fig)})
        else:
            wandb.log({tag: wandb.Image(fig)}, step=step)

    plt.close(fig)
