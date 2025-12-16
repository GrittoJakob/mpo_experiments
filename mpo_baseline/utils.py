import torch
import os
import gymnasium as gym

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


def gaussian_kl_diag(mu_i, mu, Ai, A, eps=1e-8):
    """
    mu_i, mu: (B, da)
    Ai, A: Cholesky-Faktoren (B, da, da) - diagonal 
    Returns:
        C_mu_dim_mean: (da,)  mean-KL pro Dim über Batch gemittelt
        C_sigma_dim_mean: (da,) var-KL pro Dim über Batch gemittelt
        C_mu_scalar, C_sigma_scalar optional
    """
    if Ai.dim() == 3:
        std_i = torch.diagonal(Ai, dim1=-2, dim2=-1)
    else:
        std_i = Ai

    if A.dim() == 3:
        std = torch.diagonal(A, dim1=-2, dim2=-1)
    else:
        std = A

    var_i = std_i**2
    var   = std**2

    # Mean-KL pro Dim
    C_mu_dim = 0.5 * ((mu - mu_i)**2) / (var_i + eps)  # (B, da)

    # Var-KL pro Dim
    C_sigma_dim = 0.5 * ( (var_i / (var + eps)) - 1.0 + torch.log((var + eps) / (var_i + eps)) )

    # Batch-Mittel -> pro Dim
    C_mu_dim_mean = C_mu_dim.mean(dim=0)         # (da,)
    C_sigma_dim_mean = C_sigma_dim.mean(dim=0)  # (da,)

    # Optional: wieder Skalar wie vorher
    C_mu_scalar = C_mu_dim_mean.sum()
    C_sigma_scalar = C_sigma_dim_mean.sum()

    return C_mu_dim_mean, C_sigma_dim_mean, C_mu_scalar, C_sigma_scalar

def limit_threads(n: int):
    # PyTorch threads
    torch.set_num_threads(n)
    torch.set_num_interop_threads(1)

    # NumPy / BLAS threads (muss vor Imports passieren, aber auch so meist ok)
    os.environ["OMP_NUM_THREADS"] = str(n)
    os.environ["OPENBLAS_NUM_THREADS"] = str(n)
    os.environ["MKL_NUM_THREADS"] = str(n)
    os.environ["NUMEXPR_NUM_THREADS"] = str(n)


def make_env(env_id, capture_video, run_name, name_prefix="rollout"):
    if capture_video:
        env = gym.make(env_id, render_mode="rgb_array")

        # in dieser (frisch erzeugten) video-env: genau Episode 0 aufnehmen
        env = gym.wrappers.RecordVideo(
            env,
            f"videos/{run_name}",
            name_prefix=name_prefix,
            episode_trigger=lambda ep: ep == 0,
        )
    else:
        env = gym.make(env_id)

    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = gym.wrappers.ClipAction(env)
    return env