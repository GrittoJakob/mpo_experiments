import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
"""
Changes:

Actor:
1. geänderte Input variables (hidden_size, oberservation_space, action_space)
2. Dist Funktion um Normal distribution auszugeben über actions:
    Normal(actor_mean,std) -> Actor_mean aus NN und std über trainierbare parameter 
3. Anpassen der Funktionen action, get_action_prob, evaluate_action_
    -outputs anpassen auf log_prob (density der Normal-Dist), action_distribution statt discrete policy

Critic:
1. geänderte Input variables (hidden_size, oberservation_space, action_space)
2. Geänderte Input dimension for continuous action: observation.shape + action.shape
    Sonst nur V-Funktion statt Q-Funktion

Fragen:
1. Full Covarianzmatrix by Cholesky factors for correlations or only diagonal covariance matrix (independant Normals)?

"""
class ContinuousActor(nn.Module):

    def __init__(self, env, hidden_size):
        super(ContinuousActor, self).__init__()
        self.env = env
        self.hidden_size = hidden_size
        self.dim_obs = env.observation_space.shape[0]
        self.dim_act = env.action_space.shape[0]

        self.actor_mean = nn.Sequential(
            nn.Linear(self.dim_obs, self.hidden_size),
            nn.Tanh(),
            nn.Linear(self.hidden_size,self.hidden_size),
            nn.Tanh(),
            nn.Linear(self.hidden_size, self.dim_act)
            )
        self.actor_logstd = nn.Parameter(torch.zeros(self.dim_act))

    def forward(self, state):           # Means
        return self.actor_mean(state) 


    def dist(self, state):              # Normal (Mean, std)
        action_mean = self.forward(state)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        action_distribution = Normal(action_mean, action_std) 
        return action_distribution
    
    def action(self, state):
        with torch.no_grad():
            action_distribution = self.dist(state)
            action = action_distribution.sample()
            log_probs = action_distribution.log_prob(action).sum(1)
        return action, log_probs
    
    def get_action_prob(self, state):
        action_distribution = self.dist(state) 
        log_probs = action_distribution.log_prob(action).sum(1)
        return log_probs
    
    def evaluate_action(self, state, action):
        action_distribution = self.dist(state)
        log_probs = action_distribution.log_prob(action).sum(1)
        entropy = action_distribution.entropy().sum(1)
        return action_distribution, log_prob, entropy


class Critic(nn.Module):
    def __init__(self, env, hidden_size=256):
        super(Critic, self).__init__()
        self.dim_obs = env.observation_space.shape[0]
        self.dim_act = env.action_space.shape[0]
        self.critic_net = nn.Sequential(
            nn.Linear(self.dim_obs + self.dim_act, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size,hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1))
    
    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)   # (batch, obs+act)
        return self.critic_net(x)    
      

    