try:
    import wandb
except ImportError:
    wandb = None


def _wandb_is_active():
    return wandb is not None and getattr(wandb, "run", None) is not None


def _to_python_number(value):
    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, RuntimeError):
            pass
    return value


def logging_wandb(writer, replaybuffer, stats_m_step, stats_e_step, critic_update_stats, grad_updates, num_steps):
    
    # Compute mean reward/return in replay buffer
    mean_reward_buffer = replaybuffer.mean_reward()

    metrics = {
        "grad_updates": grad_updates,
        "buffer/size": len(replaybuffer),
        "buffer/mean_reward_per_step": mean_reward_buffer,
        "buffer/total_num_steps": num_steps,
        "m-step/loss_p": stats_m_step["loss_p"],
        "m-step/loss_l": stats_m_step["loss_l"],
        "m-step/C_mu_mean": stats_m_step["C_mu_mean"],
        "m-step/C_sigma_mean": stats_m_step["C_sigma_mean"],
        "m-step/mu_mean": stats_m_step["mu_mean"],
        "m-step/std_mean": stats_m_step["std_mean"],
        "m-step/eta_mu": stats_m_step["eta_mu"],
        "m-step/eta_sigma": stats_m_step["eta_sigma"],
        "e-step/eta_dual": stats_e_step["eta_dual"],
        "e-step/loss_dual": stats_e_step["loss_dual"],
        "critic_update/q_loss": critic_update_stats["critic_loss"],
        "critic_update/q_current_mean": critic_update_stats["q_current_mean"],
        "critic_update/q_target_mean": critic_update_stats["q_target_mean"],
    }

    if "eta_penalty" in stats_e_step:
        metrics["e-step/eta_penalty"] = stats_e_step["eta_penalty"]
        metrics["e-step/loss_penalty"] = stats_e_step["loss_penalty"]

    if writer is not None:
        for key, value in metrics.items():
            if key == "grad_updates":
                continue
            writer.add_scalar(key, _to_python_number(value), grad_updates)
        writer.flush()

    if _wandb_is_active():
        wandb.log(
            {key: _to_python_number(value) for key, value in metrics.items()},
            step=grad_updates,
        )
