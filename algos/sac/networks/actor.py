import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.normal import Normal
import numpy as np
from torch.distributions import MultivariateNormal


class ActorNetwork(nn.Module):
    def __init__(self, 
                 alpha, 
                 input_dims, 
                 action_std_init,
                 n_actions=2,
                 fc1_dims=128, 
                 fc2_dims=64, 
                 fc3_dims=32, 
                 name='actor', 
                 chkpt_dir='chkpt/') -> None:
        super(ActorNetwork, self).__init__()
        self.name = name
        self.beta = alpha

        # layer dims
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.fc3_dims = fc3_dims
        self.n_actions = n_actions
 
        # Chkpt directory and filename
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, self.name+'_sac')

        self.reparam_noise = 1e-6
        
        self.code_size = 95
        self.nav_size = 6
        
        self.cov_var = torch.full((self.n_actions,), action_std_init)

        # Create the covariance matrix
        self.cov_mat = torch.diag(self.cov_var).unsqueeze(dim=0)

        # # Model layers
        # self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        # self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        # self.fc3 = nn.Linear(self.fc2_dims, self.fc3_dims)
        # self.mu = nn.Linear(self.fc3_dims, self.n_actions) # mean
        # self.sigma = nn.Linear(self.fc3_dims, self.n_actions) # std
        
        self.reshape = nn.Sequential(
                        nn.Linear(2048, self.code_size),
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
                        nn.Linear(32, self.n_actions),
                        nn.Tanh()
                    )

        # initialize the optimizer with LR
        self.opt = optim.Adam(self.parameters(), lr=alpha)

        self.device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def set_action_std(self, new_action_std):
        self.cov_var = torch.full((self.n_actions,), new_action_std)
    
    def forward(self):        
        # input = state
        # x = self.fc1(input)
        # x = F.tanh(x)
        # x = self.fc2(x)
        # x = F.tanh(x)
        # x = self.fc3(x)
        # x = F.tanh(x)
        # mu = self.mu(x)
        # sigma = self.sigma(x)
        # sigma = torch.clamp(sigma, min=self.reparam_noise, max=1) # clamp is computationally better than sigmoid
        # return mu, sigma
        raise NotImplementedError
    
    def sample_normal(self, obs, reparameterize=True):
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

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        chkpt = torch.load(self.checkpoint_file)
        torch.load_state_dict(chkpt)













