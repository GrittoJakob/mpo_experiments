import torch

    
def expectation_step(self, target_q, sampled_actions, collect_stats: bool=False):
    """
    E-step of MPO:
    - Sample actions from the target policy for each state.
    - Evaluate Q-values with the target critic.
    - Update the temperature parameter eta via the dual function.
    - Compute normalized weights over actions (norm_target_q).
    """
    diff_out_of_bound = None
    
    if self.eta_dual.grad is not None:
        self.eta_dual.grad = None
    if self.use_action_penalty and self.eta_penalty.grad is not None:
        self.eta_penalty.grad = None

    # Dual function returns a scalar that we differentiate w.r.t. eta
    norm_target_q, loss_temperature = self.compute_weights_temperature_loss(self.eta_dual, target_q, self.eps_dual)
    loss_dual = loss_temperature

    has_oob = None
    penalty_normalized_weights = None
    loss_penalty_temperature = None
    diff_out_of_bound = None
    
    stats = None
    if collect_stats:
        stats = {
            "eta_dual": self.eta_dual.detach(),
            "norm_target_q_mean": norm_target_q.detach().mean(),
            "norm_target_q_min":  norm_target_q.detach().amin(),
            "norm_target_q_max":  norm_target_q.detach().amax(),
            "loss_dual":          loss_dual.detach(),
        }

    if self.use_action_penalty:
        # Compute action penalty when out of bound:
        actions_squashed  = torch.max(torch.min(sampled_actions, self.action_space_high), self.action_space_low)
        diff_out_of_bound = sampled_actions - actions_squashed
        has_oob = (diff_out_of_bound.abs().amax() > 0).to(norm_target_q.dtype)
        
        cost_out_of_bound = -(diff_out_of_bound.pow(2).sum(dim=-1))  # (N,B)
        penalty_normalized_weights, loss_penalty_temperature = self.compute_weights_temperature_loss(
            self.eta_penalty, cost_out_of_bound, self.eps_penalty
            )

        lam_eff = self.lam * has_oob

        norm_target_q = (1 - lam_eff) * norm_target_q + lam_eff * penalty_normalized_weights
        norm_target_q = norm_target_q / (norm_target_q.sum(dim=0, keepdim=True) + 1e-8)
        loss_dual = (1 - lam_eff) * loss_dual + lam_eff * loss_penalty_temperature
        

    loss_dual.backward()

    with torch.no_grad():
        # Gradient descent update on eta
        self.eta_dual.add_(-self.eta_lr * self.eta_dual.grad)
        self.eta_dual.clamp_(min=1e-3)
        self.eta_dual.grad = None
        
        if self.use_action_penalty:
            # if has_oob=0, grad is effective 0 cause lam_eff=0 -> no Update
            self.eta_penalty.add_(-self.eta_penalty_lr * self.eta_penalty.grad)
            self.eta_penalty.clamp_(min=1e-3)
            self.eta_penalty.grad = None



        if self.use_action_penalty and collect_stats:
            stats.update({
                "eta_penalty": self.eta_penalty.detach(),
                "has_out_of_bound": has_oob.detach() if has_oob is not None else None,
                "diff_out_abs_mean": diff_out_of_bound.detach().abs().mean(),
                "diff_out_abs_max":  diff_out_of_bound.detach().abs().amax(),
                "penalty_weights_mean": penalty_normalized_weights.detach().mean(),
                "penalty_weights_min":  penalty_normalized_weights.detach().amin(),
                "penalty_weights_max":  penalty_normalized_weights.detach().amax(),
                "norm_weights_mean":  norm_target_q.detach().mean(),
                "norm_weights_min":  norm_target_q.detach().amin(),
                "norm_weights_max":  norm_target_q.detach().amax(),
                "loss_penalty":         loss_penalty_temperature.detach() if loss_penalty_temperature is not None else None,
            })

    return norm_target_q, stats


def compute_weights_temperature_loss(self, temperature: torch.Tensor, values: torch.Tensor, epsilon: float):
    """
    temperature: torch scalar, requires_grad=True
    values: (N, B)  (keine Gradienten; typischerweise aus target critic, detached)
    epsilon: float
    returns:
    weights: (N, B) detached
    dual_loss: scalar tensor (hat Grad wrt temperature)
    """
    values = values.detach()  # stop grad for 

    # weights (stop grad wrt temperature) 
    tempered = values / temperature.detach()
    tempered = tempered - tempered.max(dim=0, keepdim=True).values  # stable
    weights = torch.softmax(tempered, dim=0).detach()  # detach == stop-gradient

    # dual loss (Grad wrt temperature) 
    values_T = values.transpose(0, 1)  # (B, N)
    max_v = values_T.max(dim=1, keepdim=True).values  # (B, 1)
    exp_term = torch.exp((values_T - max_v) / temperature)  # grad fließt in temperature
    log_mean_exp = torch.log(exp_term.mean(dim=1) + 1e-8)  # (B,)

    dual_loss = temperature * epsilon + max_v.mean() + temperature * log_mean_exp.mean()


    return weights, dual_loss