import torch.nn.functional as F
import torch.nn as nn
import torch
from torch.distributions import MultivariateNormal

class Actor(nn.Module):
    
    #Policy network
    
    def __init__(self, env, hidden_size_actor):
        super(Actor, self).__init__()
        self.env = env
        self.ds = env.observation_space.shape[0]
        self.da = env.action_space.shape[0]
        self.hs= hidden_size_actor

        self.backbone = nn.Sequential(
            nn.Linear(self.ds, self.hs),
            nn.ELU(),
            nn.Linear(self.hs, self.hs),
            nn.ELU(),
        )

        # zwei getrennte Köpfe
        self.mean_layer = nn.Linear(256, self.da)
        self.cholesky_layer = nn.Linear(256, (self.da * (self.da + 1)) // 2)


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
        if torch.isnan(state).any() or torch.isinf(state).any():
            print("🚨 NaN/Inf im INPUT state:", state)
            raise ValueError("NaN/Inf in state (Actor forward)")

        # Do i need normalization of action?
        #action_low = torch.from_numpy(self.env.action_space.low)[None, ...].to(device)  # (1, da)
        #action_high = torch.from_numpy(self.env.action_space.high)[None, ...].to(device)  # (1, da)
        #action_low = torch.as_tensor(self.env.action_space.low, device=device, dtype=torch.float32).unsqueeze(0)
        #action_high = torch.as_tensor(self.env.action_space.high, device=device, dtype=torch.float32).unsqueeze(0)
        x = self.backbone(state)   # (B, 256)
        
        #Mean Kopf
        mean = self.mean_layer(x)   # (B, da)
        #mean = action_low + (action_high - action_low) * mean

        # Debug
        if torch.isnan(mean).any():
            print("⚠️ NaN in Actor mean! state:", state)
            raise ValueError("NaN in Actor mean")


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
        return mean, cholesky
    """
    def distribution(mean, cholesky):

        action_distribution = MulivariateNormal(mean, scale_tril = cholesky)
        return action_distribution
    """    
    def action(self, state):
        """
        :param state: (ds,)
        :return: an action
        """
        with torch.no_grad():
            state_batched = state.unsqueeze(0)
            mean, cholesky = self.forward(state_batched)
            action_distribution = MultivariateNormal(mean, scale_tril=cholesky)
            action = action_distribution.sample()
        return action[0]
