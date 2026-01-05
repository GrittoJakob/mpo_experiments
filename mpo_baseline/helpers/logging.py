
import os
import wandb


def log_callback(logs):

    global_update = logs["global_update"]

    # --- TensorBoard ---
    for key in [
        "iteration",
        "global_update",
        "mean_return_buffer",
        "mean_reward_buffer",
        "mean_loss_q",
        "mean_loss_p",
        "mean_loss_l",
        "mean_current_q",
        "mean_target_q",
        "eta",
        "runtime_env",
        "runtime_eval",
        "runtime_policy_eval",
        "runtime_M_step",
        "runtime_E_step",
        "max_kl_sigma",
        "mean_sigma_det",
        "eta_mu_mean",
        "eta_mu_max",
        "eta_mu_min",
        "eta_sigma_mean",
        "eta_sigma_max",
        "eta_sigma_min",
        "return_eval",
        "var_mean",
        "var_min",
        "var_max"
        "buffer_size",
        "min_kl_sigma",
        "min_kl_mu"
    ]:
        if key in logs:
            tag = key.replace("_", "/") if key.startswith("eval_") else key
            writer.add_scalar(tag, logs[key], global_update)
                # --- W&B ---
    wandb.log(logs, step=global_update)