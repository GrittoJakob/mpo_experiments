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

def limit_threads(n: int):
    # PyTorch threads
    torch.set_num_threads(n)
    torch.set_num_interop_threads(1)

    # NumPy / BLAS threads (muss vor Imports passieren, aber auch so meist ok)
    os.environ["OMP_NUM_THREADS"] = str(n)
    os.environ["OPENBLAS_NUM_THREADS"] = str(n)
    os.environ["MKL_NUM_THREADS"] = str(n)
    os.environ["NUMEXPR_NUM_THREADS"] = str(n)


def make_env(env_id, capture_video, run_name):
    if capture_video:
        env = gym.make(env_id, render_mode="rgb_array")
        env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
    else:
        env = gym.make(env_id)

    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = gym.wrappers.ClipAction(env)
    return env


    