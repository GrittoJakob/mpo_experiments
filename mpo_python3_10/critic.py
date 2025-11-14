import torch
import torch.nn.functional as F
import torch.nn as nn


class Critic(nn.Module):
    """
    :param env: OpenAI gym environment
    """
    def __init__(self, env):
        super(Critic, self).__init__()
        self.env = env
        self.ds = env.observation_space.shape[0]
        self.da = env.action_space.shape[0]

        self.net = nn.Sequential(
            nn.Linear(self.ds + self.da, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
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
