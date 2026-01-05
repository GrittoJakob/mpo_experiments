import torch
import torch.nn as nn


class Critic(nn.Module):
    """
    State-action value function Q(s, a) approximator.

    This critic takes as input a state and an action, concatenates them,
    and outputs a single scalar Q-value per batch element.
    """
    def __init__(self, args):

        """
        :param env: Gym/Gymnasium-like environment (used to infer state/action dims)
        :param hidden_size_critic: hidden layer size of the critic network
        """
        super(Critic, self).__init__()

        # Dimensions
        self.ds = args.obs_space
        self.da = args.action_dim
        self.hs = args.hidden_size_critic

        # Simple MLP over concatenated [state, action]
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
        Forward pass of the critic network.

        :param state:  Tensor of shape (B, ds)
        :param action: Tensor of shape (B, da)
        :return: Q-values with shape (B, 1)
        """

        # Concatenate state and action along the feature dimension
        h = torch.cat([state, action], dim=1)  # (B, ds+da)

        # Pass through the network
        Q = self.net(h)     #(B,1)
        return Q
