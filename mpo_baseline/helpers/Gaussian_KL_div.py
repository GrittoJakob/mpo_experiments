import torch
import os
import gymnasium as gym
import wandb
def gaussian_kl_diag(mu_off, mu_on, std_off, std_on, use_mass_force_KL, eps: float = 1e-8):
    """
    mu/std: [B, A] oder allgemein [..., A]
    Returns: (A,), (A,), scalar, scalar
    """nnnnn
    std_off = std_off.clamp_min(eps)
    std_on  = std_on.clamp_min(eps)


    reduce_dims = tuple(range(mu_on.dim() - 1))

    if use_mass_force_KL:
        # C_mu: 0.5 * (mu_off - mu_on)^2 / std_on^2
        inv_std_on = std_on.reciprocal()
        d = (mu_off - mu_on) * inv_std_on
        C_mu = 0.5 * (d * d)

        # C_sigma: log(std_on/std_off) + 0.5*( (std_off/std_on)^2 - 1 )
        r = std_off * inv_std_on
        C_sigma = (std_on.log() - std_off.log()) + 0.5 * (r * r - 1.0)
    else:
        # C_mu: 0.5 * (mu_on - mu_off)^2 / std_off^2
        inv_std_off = std_off.reciprocal()
        d = (mu_on - mu_off) * inv_std_off
        C_mu = 0.5 * (d * d)

        # C_sigma: log(std_off/std_on) + 0.5*( (std_on/std_off)^2 - 1 )
        r = std_on * inv_std_off
        C_sigma = (std_off.log() - std_on.log()) + 0.5 * (r * r - 1.0)

    C_mu_dim_mean    = C_mu.mean(dim=reduce_dims)       # [A]
    C_sigma_dim_mean = C_sigma.mean(dim=reduce_dims)    # [A]

    return C_mu_dim_mean, C_sigma_dim_mean, C_mu_dim_mean.sum(), C_sigma_dim_mean.sum()