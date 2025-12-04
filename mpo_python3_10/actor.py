import torch.nn.functional as F
import torch.nn as nn
import torch
from torch.distributions import MultivariateNormal

class Actor(nn.Module):
    
    #Policy network
    
    def __init__(self, env, hidden_size_actor, std_init):
        super(Actor, self).__init__()
        self.env = env
        self.ds = env.observation_space.shape[0]
        self.da = env.action_space.shape[0]
        self.hs= hidden_size_actor
        self._printed_init_cov = False
        

        self.backbone = nn.Sequential(
            nn.Linear(self.ds, self.hs),
            nn.LayerNorm(self.hs),
            nn.Tanh(),
            nn.Linear(self.hs, self.hs),
            nn.ELU(),
        )

        # zwei getrennte Köpfe
        self.mean_layer = nn.Linear(self.hs, self.da)
        self.cholesky_layer = nn.Linear(self.hs, (self.da * (self.da + 1)) // 2)
        
        with torch.no_grad():
            # all weiths to zero
            self.cholesky_layer.weight.zero_()
            self.cholesky_layer.bias.zero_()

            # inverse Softplus:
            def softplus_inv(y):
                return torch.log(torch.exp(torch.tensor(y)) - 1.0)

            diag_indices = torch.arange(self.da, dtype=torch.long)
            diag_indices = (diag_indices + 1) * (diag_indices + 2) // 2 - 1

            # Bias so setzen, dass softplus(bias) = std_init
            self.cholesky_layer.bias[diag_indices] = softplus_inv(std_init)


    def forward(self, state):
        """
        forwards input through the network
        :param state: (B, ds)
        :return: mean vector (B, da) and cholesky factorization of covariance matrix (B, da, da)
        """
        device = state.device
        B = state.size(0)
        ds = self.ds
        da = self.da

        x = self.backbone(state)   # (B, hs)
        
        #Mean Kopf
        mean = self.mean_layer(x)   # (B, da)
    
        # Cholesky-Kopf
        cholesky_vector = self.cholesky_layer(x)   # (B, da*(da+1)//2)

        cholesky_diag_index = torch.arange(da, dtype=torch.long, device=device) + 1
        cholesky_diag_index = (cholesky_diag_index * (cholesky_diag_index + 1)) // 2 - 1
        cholesky_vector[:, cholesky_diag_index] = F.softplus(
            cholesky_vector[:, cholesky_diag_index]
        )

        tril_indices = torch.tril_indices(row=da, col=da, offset=0, device=device)
        cholesky = torch.zeros(size=(B, da, da), dtype=torch.float32, device=device)
        cholesky[:, tril_indices[0], tril_indices[1]] = cholesky_vector

        #Debug: Print variance
        if not self._printed_init_cov:
            with torch.no_grad():
                Sigma = cholesky @ cholesky.transpose(-1, -2)   # Covariance
                diag_vars = torch.diagonal(Sigma, dim1=-2, dim2=-1)[0]  # erste Batch-Zeile
                print("🔍 Initial diagonal variances:", diag_vars.cpu().numpy())
            self._printed_init_cov = True

        return mean, cholesky
   
     
    def action(self, state, clip_to_env: bool = True):
        """
        :param state: (ds,)
        :return: an action
        """
        with torch.no_grad():
            state_batched = self.ensure_batched(state)
            mean, cholesky = self.forward(state_batched)
            action_distribution = MultivariateNormal(mean, scale_tril=cholesky)
            action = action_distribution.sample()[0]

            if clip_to_env:
                low = torch.as_tensor(self.env.action_space.low, device=action.device, dtype=action.dtype)
                high = torch.as_tensor(self.env.action_space.high, device=action.device, dtype=action.dtype)
                action = torch.clamp(action, low, high)
        return action.cpu().numpy()

    def evaluate_action(self, state, action):
        """
        Compute log-probability of given action under the actor's policy.
        :param state: (ds,) or (B, ds)
        :param action: (da,) or (B, da)
        :return: log_prob tensor with shape (B,)
        """
        state_batched = self.ensure_batched(state)
        action_batched = self.ensure_batched(action)
        mean, cholesky = self.forward(state_batched)
        action_distribution = MultivariateNormal(mean, scale_tril=cholesky)
        log_prob = action_distribution.log_prob(action_batched)
        return log_prob

    def sample_action(self, state, sample_num):
        """
        Sample multiple actions per state from the current policy.
        :param state: (ds,) or (B, ds)
        :param sample_num: number of action samples per state (N)
        :return: samples with shape (B, N, da)
        """
        with torch.no_grad():
            state_batched = self.ensure_batched(state)
            mean, cholesky = self.forward(state_batched)
            dist = MultivariateNormal(mean, scale_tril=cholesky)
            samples = dist.rsample((sample_num,)).permute(1, 0, 2)
        return samples

    def ensure_batched(self,tensor):
        return tensor if tensor.ndim == 2 else tensor.unsqueeze(0)