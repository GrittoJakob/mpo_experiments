import torch
import matplotlib.pyplot as plt


def plot_mpo_weight_distribution(
    sampled_actions: torch.Tensor,
    weights: torch.Tensor,
    b_idx: int = 0,
    title: str = "",
    action_low: torch.Tensor = None,
    action_high: torch.Tensor = None,
    sort_x: bool = True,
):
    """
    Plottet für ein bestimmtes Batch-Sample die erste und letzte Action-Dimension
    gegen die zugehörigen MPO-Weights.

    Args:
        sampled_actions: Tensor mit Shape (N, B, A)
        weights: Tensor mit Shape (N, B)
        b_idx: Batch-Index des Samples (bei batch_size=1 -> 0)
        title: Titel des Plots
        action_low: optional, Tensor mit Shape (A,) oder broadcastbar
        action_high: optional, Tensor mit Shape (A,) oder broadcastbar
        sort_x: sortiert x-Werte für lesbarere Linienplots
    """
    assert sampled_actions.ndim == 3, f"expected sampled_actions shape (N,B,A), got {sampled_actions.shape}"
    assert weights.ndim == 2, f"expected weights shape (N,B), got {weights.shape}"
    assert sampled_actions.shape[:2] == weights.shape, (
        f"sampled_actions.shape[:2]={sampled_actions.shape[:2]} must match weights.shape={weights.shape}"
    )
    assert 0 <= b_idx < sampled_actions.shape[1], f"b_idx={b_idx} out of range for B={sampled_actions.shape[1]}"

    with torch.no_grad():
        actions_np = sampled_actions.detach().cpu()
        weights_np = weights.detach().cpu()

        x_first = actions_np[:, b_idx, 0].numpy()      # (N,)
        x_last  = actions_np[:, b_idx, -1].numpy()     # (N,)
        y       = weights_np[:, b_idx].numpy()         # (N,)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        fig.suptitle(title if title else f"MPO weight distribution for batch sample {b_idx}")

        for ax, x, dim_name, dim_idx in [
            (axes[0], x_first, "erste Action-Dimension", 0),
            (axes[1], x_last,  "letzte Action-Dimension", sampled_actions.shape[-1] - 1),
        ]:
            if sort_x:
                order = x.argsort()
                x_plot = x[order]
                y_plot = y[order]
            else:
                x_plot = x
                y_plot = y

            ax.scatter(x_plot, y_plot, alpha=0.8, s=20)
            ax.plot(x_plot, y_plot, alpha=0.6)

            if action_low is not None:
                low_val = action_low[dim_idx].detach().cpu().item() if torch.is_tensor(action_low) else action_low[dim_idx]
                ax.axvline(low_val, linestyle="--", alpha=0.7, label="action_low")

            if action_high is not None:
                high_val = action_high[dim_idx].detach().cpu().item() if torch.is_tensor(action_high) else action_high[dim_idx]
                ax.axvline(high_val, linestyle="--", alpha=0.7, label="action_high")

            ax.set_xlabel(f"Action value ({dim_name})")
            ax.set_ylabel("Weight")
            ax.set_title(dim_name)
            ax.grid(True, alpha=0.3)

            handles, labels = ax.get_legend_handles_labels()
            if labels:
                ax.legend()

        plt.tight_layout()
        plt.show()