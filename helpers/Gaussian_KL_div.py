import torch

def gaussian_kl_diag(mu_1, std_1, mu_2, std_2, eps: float = 1e-8):

    # Clamp std for numerical safety
    std_1 = std_1.clamp_min(eps)
    std_2 = std_2.clamp_min(eps)


    reduce_dims = tuple(range(mu_1.dim() - 1))

    inv_std_2 = std_2.reciprocal()
    d1 = (mu_1 - mu_2) * inv_std_2
    d2 = std_1 * inv_std_2

    KL_div = torch.log(std_2/std_1) + 0.5*( d1*d1  + d2*d2 - 1)
    KL_div_mean = KL_div.mean(dim= reduce_dims)


    return KL_div_mean