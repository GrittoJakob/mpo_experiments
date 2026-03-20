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
        self.dim_states = args.obs_dim
        self.dim_action = args.action_dim
        self.hidden_size = args.hidden_size_critic

        # Simple MLP over concatenated [state, action]
        self.net = nn.Sequential(
            nn.Linear(self.dim_states + self.dim_action, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.Tanh(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ELU(),
            nn.Linear(self.hidden_size, 1)
        )
   
    def forward(self, state, action):
        """
        Forward pass of the critic network.

        :param state:  Tensor of shape (B, dim_states)
        :param action: Tensor of shape (B, dim_action)
        :return: Q-values with shape (B, 1)
        """

        # Concatenate state and action along the feature dimension
        h = torch.cat([state, action], dim=1)  # (B, dim_states+dim_action)

        # Pass through the network
        Q = self.net(h)     #(B,1)
        return Q
