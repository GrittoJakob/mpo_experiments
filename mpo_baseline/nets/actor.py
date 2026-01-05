import torch.nn.functional as F
import torch.nn as nn
import torch
from torch.distributions import Independent, Normal

class Actor(nn.Module):
    
    #Policy network
    
    def __init__(self, args):
        super(Actor, self).__init__()
        self.ds = args.obs_space
        self.da = args.action_dim
        self.hs= args.hidden_size_actor
        self._printed_init_cov = False
        self.use_state_dependent_var = args.use_state_dependent_var
        std_init = args.std_init
        

        self.backbone = nn.Sequential(
            nn.Linear(self.ds, self.hs),
            nn.LayerNorm(self.hs),
            nn.Tanh(),
            nn.Linear(self.hs, self.hs),
            nn.ELU(),
        )

        # zwei getrennte Köpfe
        self.mean_layer = nn.Linear(self.hs, self.da)
               # inverse Softplus helper
        def softplus_inv(y):
            return torch.log(torch.exp(torch.tensor(y)) - 1.0)

        if self.use_state_dependent_var:

            self.std_layer = nn.Linear(self.hs, self.da)
            with torch.no_grad():
                self.std_layer.weight.zero_()
                self.std_layer.bias.fill_(softplus_inv(std_init))
        else:
            self.log_std = nn.Parameter(torch.ones(self.da) * -0.5) # -1 scaling to make initialization calmer

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
        
        if self.use_state_dependent_var:
            std = F.softplus(self.std_layer(x))    # (B, da)
        
        else:
            std = std.log_std

        #Debug: Print variance
        if not self._printed_init_cov:
            with torch.no_grad():
                print("🔍 Initial diagonal variances:", std.cpu().numpy())
            self._printed_init_cov = True

        return mean, std
   
     
    def action(self, state, clip_to_env: bool = False, deterministic: bool = False):
        """
        :param state: (ds,)
        :return: an action
        """
        with torch.no_grad():
            state_batched = self.ensure_batched(state)
            mean, std = self.forward(state_batched)
            action_distribution = Independent(Normal(mean, std), 1)

            if deterministic:
                action = mean[0]
            else:
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
        mean, std= self.forward(state_batched)
        action_distribution = Independent(Normal(mean, std), 1)
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
            mean, std = self.forward(state_batched)
            dist = Independent(Normal(mean,std), 1)
            samples = dist.rsample((sample_num,)).permute(1, 0, 2)
        return samples, mean, std


    def ensure_batched(self,tensor):
        return tensor if tensor.ndim == 2 else tensor.unsqueeze(0)