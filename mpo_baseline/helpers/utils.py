import torch
import os
import gymnasium as gym
import wandb

def btr(m):
    return m.diagonal(dim1=-2, dim2=-1).sum(-1)

def bt(m):
    return m.transpose(dim0=-2, dim1=-1)


def gaussian_kl(mu_i, mu, Ai, A):
    """
       Computes KL for multivariate Gaussians with Cholesky factors Ai (old) and A (new).
    Returns:
      - C_mu:    mean KL for mean
      - C_sigma: mean KL for covariance
      - sigma_i_det: mean determinant of old cov
      - sigma_det: mean determinant of new cov
      - var_mean: mean diagonal variance of Σ
      - var_min:  minimum diagonal variance
      - var_max:  maximum diagonal variancen: mean of determinanats of sigma_i, sigma

    """
    n = A.size(-1)
    mu_i = mu_i.unsqueeze(-1)  # (B, n, 1)
    mu = mu.unsqueeze(-1)  # (B, n, 1)

    sigma_i = Ai @ bt(Ai)  # (B, n, n)
    sigma = A @ bt(A)  # (B, n, n)
    sigma_i_det = sigma_i.det()  # (B,)
    sigma_det = sigma.det()  # (B,)
    sigma_i_det = torch.clamp_min(sigma_i_det, 1e-6)
    sigma_det = torch.clamp_min(sigma_det, 1e-6)
    sigma_i_inv = sigma_i.inverse()  # (B, n, n)
    sigma_inv = sigma.inverse()  # (B, n, n)

    inner_mu = ((mu - mu_i).transpose(-2, -1) @ sigma_i_inv @ (mu - mu_i)).squeeze()  # (B,)
    inner_sigma = torch.log(sigma_det / sigma_i_det) - n + btr(sigma_inv @ sigma_i)  # (B,)
    C_mu = 0.5 * torch.mean(inner_mu)
    C_sigma = 0.5 * torch.mean(inner_sigma)
    
    var_diag = torch.diagonal(sigma, dim1=-2, dim2=-1)  # (B,n)
    var_mean = var_diag.mean()         # Mean 
    var_min  = var_diag.min()          # Minimal variance
    var_max  = var_diag.max()          # maximum variance
    return C_mu, C_sigma, torch.mean(sigma_i_det), torch.mean(sigma_det), var_mean, var_min, var_max


def gaussian_kl_diag(mu_q, mu, A_q, A, eps=1e-8, use ):
    """
    mu_i, mu: (B, da)
    Ai, A: Cholesky-Faktoren (B, da, da) - diagonal 
    Returns:
        C_mu_dim_mean: (da,)  mean-KL pro Dim über Batch gemittelt
        C_sigma_dim_mean: (da,) var-KL pro Dim über Batch gemittelt
        C_mu_scalar, C_sigma_scalar optional
    """
    if A_q.dim() == 3:
        std_q = torch.diagonal(A_q, dim1=-2, dim2=-1)
    else:
        std_q = A_q

    if A.dim() == 3:
        std = torch.diagonal(A, dim1=-2, dim2=-1)
    else:
        std = A

    var_q = std_q**2
    var   = std**2

    if use_mass_KL:
        C_mu_dim = 0.5 * ((mu_q - mu)**2) / (var_q + eps)
        C_sigma_dim = 0.5 * ( (var_q / (var + eps)) - 1.0 + torch.log((var + eps) / (var_q + eps)) )
    
    else:
        # Mean-KL pro Dim
        C_mu_dim = 0.5 * ((mu - mu_q)**2) / (var + eps)  # (B, da)

        # Var-KL pro Dim
        C_sigma_dim = 0.5 * ( (var / (var_q + eps)) - 1.0 + torch.log((var_q + eps) / (var + eps)) )

    # Batch-Mittel -> pro Dim
    C_mu_dim_mean = C_mu_dim.mean(dim=0)         # (da,)
    C_sigma_dim_mean = C_sigma_dim.mean(dim=0)  # (da,)

    # Optional: wieder Skalar wie vorher
    C_mu_scalar = C_mu_dim_mean.sum()
    C_sigma_scalar = C_sigma_dim_mean.sum()

    return C_mu_dim_mean, C_sigma_dim_mean, C_mu_scalar, C_sigma_scalar

