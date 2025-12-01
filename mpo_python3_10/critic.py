import torch
import torch.nn.functional as F
import torch.nn as nn


class Critic(nn.Module):
    """
    :param env: OpenAI gym environment
    """
    def __init__(self, env, hidden_size_critic):
        super(Critic, self).__init__()
        self.env = env
        self.ds = env.observation_space.shape[0]
        self.da = env.action_space.shape[0]
        self.hs = hidden_size_critic

        self.net = nn.Sequential(
            nn.Linear(self.ds + self.da, self.hs),
            nn.LayerNorm(self.hs),
            nn.Tanh(),
            nn.Linear(self.hs, self.hs),
            nn.ELU(),
            nn.Linear(self.hs, 1)
        )
   
    def forward(self, state, action):
        """
        :param state: (B, ds)
        :param action: (B, da)
        :return: Q-value
        """
        h = torch.cat([state, action], dim=1)  # (B, ds+da)
        Q = self.net(h)
        return Q
