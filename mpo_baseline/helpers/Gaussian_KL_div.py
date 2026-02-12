import torch
import os
import gymnasium as gym
import wandb

def gaussian_kl_diag(mu_off, mu_on, std_off, std_on, use_mass_force_KL, eps: float = 1e-8):
    """
    KL divergence for diagonal Gaussians with arbitrary leading dimensions (e.g. [B, T, da]).

    We compute KL( N_i || N ), i.e. "old/behavior" (mu_i, Ai) relative to "new" (mu, A).

    Args:
        mu_i, mu: [..., da]               Means (old, new)
        Ai, A:   [..., da] or [..., da, da]
                 Either per-dimension stddev (last dim = da),
                 or a (lower-triangular) Cholesky matrix where only the diagonal is used.

    Returns:
        C_mu_dim_mean:    (da,) mean KL contribution from the means, averaged over all leading dims
        C_sigma_dim_mean: (da,) mean KL contribution from the variances, averaged over all leading dims
    """

    assert mu_off.shape == mu_on.shape == std_off.shape == std_on.shape
    assert mu_on.dim() >= 1
    # Already std (or diagonal) in the last dimension

    var_off = std_off ** 2  # old variance
    var_on   = std_on ** 2    # new variance


    # Mean contribution for KL(N_i || N):
    if use_mass_force_KL:
        # 0.5 * (mu_i - mu)^2 / var
        C_mu_dim = 0.5 * (mu_off - mu_on) ** 2 / (var_on + eps)  # [..., da]
        C_sigma_dim = torch.log(std_on / (std_off + eps)) + 0.5 * ((var_off / var_on + eps) - 1)  
    
    else:
        C_mu_dim = 0.5 * (mu_on - mu_off) ** 2 / (var_off + eps)  # [..., da]
        C_sigma_dim = torch.log(std_off / (std_on + eps)) + 0.5 * ((var_on / var_off + eps) - 1)  # [M, dim_act]


    # Batch-Mittel -> pro Dim
    C_mu_dim_mean = C_mu_dim.mean(dim=0)         # (da,)
    C_sigma_dim_mean = C_sigma_dim.mean(dim=0)  # (da,)

    # Optional: again scalar
    C_mu_scalar = C_mu_dim_mean.sum()
    C_sigma_scalar = C_sigma_dim_mean.sum()

    return C_mu_dim_mean, C_sigma_dim_mean, C_mu_scalar, C_sigma_scalar

