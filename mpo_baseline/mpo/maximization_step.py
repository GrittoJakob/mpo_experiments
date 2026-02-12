import torch
from torch.nn.utils import clip_grad_norm_
from torch.distributions import MultivariateNormal, Independent, Normal
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from helpers.Gaussian_KL_div import gaussian_kl, gaussian_kl_diag

def maximization_step(self, state_batch, norm_target_q, sampled_actions, mu_off, std_off, collect_stats): 
    """
    M-step of MPO:
    - Update the policy parameters to maximize the weighted log-likelihood
    under the non-parametric target distribution (defined by norm_target_q).
    - Enforce KL constraints between the new and old policy via Lagrange multipliers.
    """

    # Current actor parameters for this batch of states
    mu_on, std_on = self.actor.forward(state_batch)      

    pi_1 = Independent(Normal(mu_on, std_off), 1)
    pi_2 = Independent(Normal(mu_off, std_on), 1)
    logp1 = pi_1.log_prob(sampled_actions)
    logp2 = pi_2.log_prob(sampled_actions)

    # Weighted policy improvement objective
    weighted_logp = (norm_target_q * (logp1 + logp2)) 
    self.loss_p = - weighted_logp.sum(dim=0).mean()        # sum over N -> (B,), mean over B -> scalar

    # KL constraints between old and new Gaussian policies, options to use KL mass forcing KL divergence or zero forcing
    C_mu_dim, C_sigma_dim, _, _ = gaussian_kl_diag(mu_off, mu_on, std_off, std_on, self.use_mass_force_KL)

    with torch.no_grad():

        # Update lagrange multipliers by gradient descent
        self.eta_mu -= self.alpha_mu_scale * (self.eps_mu_dim - C_mu_dim)
        self.eta_sigma -= self.alpha_sigma_scale * (self.eps_gamma_dim - C_sigma_dim)

        # Clamp Lagrange multipliers to a reasonable range
        self.eta_mu.clamp_(1e-10, self.alpha_mu_max)
        self.eta_sigma.clamp_(1e-10, self.alpha_sigma_max)

    # Total actor loss: maximize weighted log-prob subject to KL constraints
    constraint_mu = (C_mu_dim - self.eps_mu_dim)
    constraint_sigma = (C_sigma_dim - self.eps_gamma_dim)

    self.loss_l = self.loss_p \
            + (self.eta_mu * constraint_mu).sum() \
            + (self.eta_sigma * constraint_sigma).sum()

    self.actor_optimizer.zero_grad(set_to_none=True)
    self.loss_l.backward()
    clip_grad_norm_(self.actor.parameters(), 0.1)
    self.actor_optimizer.step() 

    if collect_stats:

        # LOGGING STATS 
        C_mu_mean    = C_mu_dim.mean().detach()
        C_sigma_mean = C_sigma_dim.mean().detach()

        eta_mu_mean    = self.eta_mu.mean().detach()
        eta_mu_min     = self.eta_mu.min().detach()
        eta_mu_max     = self.eta_mu.max().detach()
        eta_sigma_mean = self.eta_sigma.mean().detach()
        eta_sigma_min  = self.eta_sigma.min().detach()
        eta_sigma_max  = self.eta_sigma.max().detach()
        std_mean       = std_on.mean().detach()
        mu_mean        = mu_on.mean().detach()
        std_dim_mean = std_on.mean(dim=(0, 1))   # -> [A]
        mu_dim_mean  = mu_on.mean(dim=(0, 1))    # -> [A]  
        std_min = std_dim_mean.min().detach()
        std_max = std_dim_mean.max().detach()

        mu_min = mu_dim_mean.min().detach()
        mu_max = mu_dim_mean.max().detach()


        stats = {
            "loss_p":        self.loss_p.detach(),
            "loss_l":        self.loss_l.detach(),
            "C_mu_mean":     C_mu_mean,
            "C_sigma_mean":  C_sigma_mean,
            "eta_mu_mean":   eta_mu_mean,
            "eta_mu_min":    eta_mu_min,
            "eta_mu_max":    eta_mu_max,
            "eta_sigma_mean":eta_sigma_mean,
            "eta_sigma_min": eta_sigma_min,
            "eta_sigma_max": eta_sigma_max,
            "std_mean":      std_mean,
            "mu_mean":       mu_mean,
            "mu_max":        mu_max,
            "mu_min":        mu_min,
            "std_max":       std_max,
            "std_min":       std_min,
        }
    else:
        stats = None

    return stats