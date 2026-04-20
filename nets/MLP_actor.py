import torch.nn.functional as F
import torch.nn as nn
import torch
from torch.distributions import Independent, Normal

class Actor(nn.Module):
    
    #Policy network
    
    def __init__(self, args):
        super(Actor, self).__init__()
        self.dim_states = args.obs_dim
        self.dim_action = args.action_dim
        self.hidden_size= args.hidden_size_actor
        self._printed_init_cov = False
        std_init = args.std_init
        self.action_space_low = args.action_space_low
        self.action_space_high = args.action_space_high
        self.use_tanh_on_mean = args.use_tanh_on_mean
        self.clip_to_env = args.clip_to_env
        
        

        self.backbone = nn.Sequential(
            nn.Linear(self.dim_states, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.Tanh(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ELU(),
        )

        # Two seperated actor heads for mean and std
        self.mean_layer = nn.Linear(self.hidden_size, self.dim_action)
        if self.use_tanh_on_mean:
            self.activation_layer_mean = nn.Tanh()
               # inverse Softplus helper
        def softplus_inv(y):
            return torch.log(torch.exp(torch.tensor(y)) - 1.0)

        self.std_layer = nn.Linear(self.hidden_size, self.dim_action)
        with torch.no_grad():
            self.std_layer.weight.zero_()
            self.std_layer.bias.fill_(softplus_inv(std_init))

    def forward(self, state):
        """
        forwardim_states input through the network
        :param state: (B, dim_states)
        :return: mean vector (B, dim_action) and diagonal covariance matrix (B, dim_action)
        B = Batch size
        """
        
        preprocessing = self.backbone(state)   # (B, hidden_size)
        
        #Mean Head
        mean = self.mean_layer(preprocessing)   # (B, dim_action)

        # Only if flag is true in input args (recommended)
        if self.use_tanh_on_mean:
            mean = self.activation_layer_mean(mean)     # (B, dim_action)
            high = torch.as_tensor(self.action_space_high, device=mean.device, dtype=mean.dtype).view(1, -1)
            assert torch.isfinite(high).all(), f"non-finite action high: {high}"
            assert high.shape[-1] == mean.shape[-1], f"high shape: {high.shape}, mean shape:  {mean.shape}"
            mean = mean * high
        # State dependant variance layer
        std = F.softplus(self.std_layer(preprocessing))    # (B, dim_action)

        return mean, std

    def get_action_distribution(self, state):

        #Ensure input is batched
        state_batched = self.ensure_batched(state)

        # Compute forward pass
        mean, std = self.forward(state_batched)
        action_distribution = Independent(Normal(mean, std), 1)

        return action_distribution, mean, std
    
     
    def action(self, state, deterministic: bool = False):
        """
        :param state: (dim_states,)
        :clip_to_env: Flag for clipping the action to the environment action bounds
        :deterministic: Flag for using deterministic action (mean) instead of sampling from the distribution, used in eval mode
        :return: an action
        """
        with torch.no_grad():
            # Get action distribution and batch status
            action_distribution, mean, _ = self.get_action_distribution(state)

            # Action sampling (deterministic in eval mode)
            if deterministic:
                action = mean
            else:
                action = action_distribution.sample()

            # Action clipping to env action bounds
            if self.clip_to_env:
                low = torch.as_tensor(self.action_space_low, device=action.device, dtype=action.dtype)
                high = torch.as_tensor(self.action_space_high, device=action.device, dtype=action.dtype)
                action = torch.clamp(action, low, high)
            
            # Ensure action dimension
            if state.ndim == 1:             # (dim_action)
                action = action.squeeze(0)


        return action.cpu().numpy()

    def sample_action(self, state, sample_num):
        """
        Sample multiple actions per state from the current policy.
        :param state: (ds,) or (B, ds)
        :param sample_num: number of action samples per state (N)
        :return: samples with shape (B, N, da) without clipping to env
        """
        with torch.no_grad():
            # Get action distribution and batch status
            action_distribution, mean, std = self.get_action_distribution(state)

            # Sample from distribution with grad on
            samples = action_distribution.rsample((sample_num,)).permute(1, 0, 2)

        return samples, mean, std

    def ensure_batched(self, state: torch.Tensor):
        if state.ndim == 1:
            return state.unsqueeze(0)
        if state.ndim == 2:
            return state
        else:
            raise ValueError(f"Expected state with ndim 1 or 2, got shape {state.shape}")
