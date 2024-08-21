import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import MultivariateNormal
from parameters import *

class ActorCritic(nn.Module):
    def __init__(self, obs_dim, action_dim, action_std_init):
        super(ActorCritic, self).__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = torch.device("cpu")
        
        # Create our variable for the matrix.
        # Note that I chose 0.2 for stdev arbitrarily.
        self.cov_var = torch.full((self.action_dim,), action_std_init)

        # Create the covariance matrix
        self.cov_mat = torch.diag(self.cov_var).unsqueeze(dim=0)

        self.code_size = 95
        self.nav_size = 6

        self.reshape = nn.Sequential(
                        nn.Linear(self.obs_dim, self.code_size),
                        nn.Tanh(),
                        nn.Linear(self.code_size, self.code_size),
                        nn.Tanh(),
                    )

        # actor
        self.actor = nn.Sequential(
                        nn.Linear(self.code_size+self.nav_size, 128),
                        nn.Tanh(),
                        nn.Linear(128, 64),
                        nn.Tanh(),
                        nn.Linear(64, 32),
                        nn.Tanh(),
                        nn.Linear(32, self.action_dim),
                        nn.Tanh()
                    )
        
        # critic
        self.critic = nn.Sequential(
                        nn.Linear(self.code_size+self.nav_size, 128),
                        nn.Tanh(),
                        nn.Linear(128, 64),
                        nn.Tanh(),
                        nn.Linear(64, 32),
                        nn.Tanh(),
                        nn.Linear(32, 1)
                    )

    def forward(self):
        raise NotImplementedError
    
    def set_action_std(self, new_action_std):
        self.cov_var = torch.full((self.action_dim,), new_action_std)
    
    def get_value(self, obs):
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float)
        obs_a = obs[:,:-self.nav_size]
        obs_n = obs[:,-self.nav_size:]
        rs_obs_a = self.reshape(obs_a)
        obs = torch.cat((rs_obs_a, obs_n), -1)
        return self.critic(obs)
    
    def get_action_and_log_prob(self, obs):
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float)
        # import pdb; pdb.set_trace()
        obs_a = obs[:,:-self.nav_size]
        obs_n = obs[:,-self.nav_size:]
        rs_obs_a = self.reshape(obs_a)
        obs = torch.cat((rs_obs_a, obs_n), -1)
        mean = self.actor(obs)
        # Create our Multivariate Normal Distribution
        dist = MultivariateNormal(mean, self.cov_mat)
        # Sample an action from the distribution and get its log prob
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.detach(), log_prob.detach()
    
    def evaluate(self, obs, action):  
        obs_a = obs[:,:-self.nav_size]
        obs_n = obs[:,-self.nav_size:]
        rs_obs_a = self.reshape(obs_a)
        obs = torch.cat((rs_obs_a, obs_n), -1)

        mean = self.actor(obs)
        cov_var = self.cov_var.expand_as(mean)
        cov_mat = torch.diag_embed(cov_var)
        dist = MultivariateNormal(mean, cov_mat)
        
        logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        values = self.critic(obs)
        
        return logprobs, values, dist_entropy


################################################################################

# class ActorCritic(nn.Module):
#     def __init__(self, obs_dim, action_dim, action_std_init, hidden_size=64, lstm_layers=1):
#         super(ActorCritic, self).__init__()
#         self.obs_dim = obs_dim
#         self.action_dim = action_dim
#         self.hidden_size = hidden_size
#         self.lstm_layers = lstm_layers
#         self.device = torch.device("cpu")
        
#         # Create our variable for the matrix.
#         self.cov_var = torch.full((self.action_dim,), action_std_init)

#         # Create the covariance matrix
#         self.cov_mat = torch.diag(self.cov_var).unsqueeze(dim=0)

#         self.code_size = 95
#         self.nav_size = 6

#         self.reshape = nn.Sequential(
#                         nn.Linear(self.obs_dim, self.code_size),
#                         nn.Tanh(),
#                         nn.Linear(self.code_size, self.code_size),
#                         nn.Tanh(),
#                     )

#         # LSTM layer
#         self.lstm = nn.LSTM(input_size=self.code_size + self.nav_size, 
#                             hidden_size=self.hidden_size, 
#                             num_layers=self.lstm_layers, 
#                             batch_first=True)

#         # actor
#         self.actor = nn.Sequential(
#                         nn.Linear(self.hidden_size, 128),
#                         nn.Tanh(),
#                         nn.Linear(128, 64),
#                         nn.Tanh(),
#                         nn.Linear(64, 32),
#                         nn.Tanh(),
#                         nn.Linear(32, self.action_dim),
#                         nn.Tanh()
#                     )
        
#         # critic
#         self.critic = nn.Sequential(
#                         nn.Linear(self.hidden_size, 128),
#                         nn.Tanh(),
#                         nn.Linear(128, 64),
#                         nn.Tanh(),
#                         nn.Linear(64, 32),
#                         nn.Tanh(),
#                         nn.Linear(32, 1)
#                     )
    
#     def forward(self):
#         raise NotImplementedError
    
#     def set_action_std(self, new_action_std):
#         self.cov_var = torch.full((self.action_dim,), new_action_std)
    
#     def get_value(self, obs):
#         if isinstance(obs, np.ndarray):
#             obs = torch.tensor(obs, dtype=torch.float)
        
#         obs_a = obs[:,:-self.nav_size]
#         obs_n = obs[:,-self.nav_size:]
#         rs_obs_a = self.reshape(obs_a)
#         obs = torch.cat((rs_obs_a, obs_n), -1)
        
#         obs, _ = self.lstm(obs.unsqueeze(0))
#         obs = obs.squeeze(0)
        
#         return self.critic(obs)
    
#     def get_action_and_log_prob(self, obs):
#         if isinstance(obs, np.ndarray):
#             obs = torch.tensor(obs, dtype=torch.float)
        
#         obs_a = obs[:,:-self.nav_size]
#         obs_n = obs[:,-self.nav_size:]
#         rs_obs_a = self.reshape(obs_a)
#         obs = torch.cat((rs_obs_a, obs_n), -1)
        
#         obs, _ = self.lstm(obs.unsqueeze(0))
#         obs = obs.squeeze(0)
        
#         mean = self.actor(obs)
        
#         # Create our Multivariate Normal Distribution
#         dist = MultivariateNormal(mean, self.cov_mat)
#         # Sample an action from the distribution and get its log prob
#         action = dist.sample()
#         log_prob = dist.log_prob(action)
#         return action.detach(), log_prob.detach()
    
#     def evaluate(self, obs, action):  
#         obs_a = obs[:,:-self.nav_size]
#         obs_n = obs[:,-self.nav_size:]
#         rs_obs_a = self.reshape(obs_a)
#         obs = torch.cat((rs_obs_a, obs_n), -1)
        
#         obs, _ = self.lstm(obs.unsqueeze(0))
#         obs = obs.squeeze(0)

#         mean = self.actor(obs)
#         cov_var = self.cov_var.expand_as(mean)
#         cov_mat = torch.diag_embed(cov_var)
#         dist = MultivariateNormal(mean, cov_mat)
        
#         logprobs = dist.log_prob(action)
#         dist_entropy = dist.entropy()
#         values = self.critic(obs)
        
#         return logprobs, values, dist_entropy