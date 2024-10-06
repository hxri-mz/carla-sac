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
        self.temp = 0.1

        # layer dims
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.fc3_dims = fc3_dims
        self.n_actions = n_actions
 
        # Chkpt directory and filename
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, self.name+'_sac')
        self.checkpoint_path = '/mnt/disks/data/carla-sac/imitation_checkpoints/actor_model_0.04560.pt'


        self.reparam_noise = 1e-6
        
        self.code_size = 95
        self.nav_size = 6
        self.max_action = 1.0
        
        self.cov_var = torch.full((self.n_actions,), 0.1)

        # Create the covariance matrix
        self.cov_mat = torch.diag(self.cov_var).unsqueeze(dim=0)
        
        self.reshape = nn.Sequential(
                        nn.Linear(2048, 1024),
                        nn.Tanh(),
                        nn.Linear(1024, 512),
                        nn.Tanh(),
                        nn.Linear(512, 256),
                        nn.Tanh(),
                        nn.Linear(256, self.code_size),
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
        
        # state_dict = torch.load(self.checkpoint_path)
        # s1 = {key.split('.')[1]+'.'+key.split('.')[2]: value for i, (key, value) in enumerate(state_dict.items()) if i < 4}
        # s2 = {key.split('.')[1]+'.'+key.split('.')[2]: value for i, (key, value) in enumerate(state_dict.items()) if i >= 4}
        
        # self.reshape.load_state_dict(s1)
        # self.actor.load_state_dict(s2)

        self.device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def set_action_std(self, new_action_std):
        self.cov_var = torch.full((self.n_actions,), new_action_std)
        # self.cov_mat = torch.diag(self.cov_var).unsqueeze(dim=0)
    
    def forward(self):        
        raise NotImplementedError
    
    def sample_normal(self, obs, reparameterize=True):
        obs_a = obs[:,:-self.nav_size]
        obs_n = obs[:,-self.nav_size:]
        rs_obs_a = self.reshape(obs_a)
        obsn = torch.cat((rs_obs_a, obs_n), -1)
        
        mean = self.actor(obsn)

        cov_mat = torch.diag(self.cov_var).unsqueeze(dim=0).repeat(obsn.size(0), 1, 1)
        dist = MultivariateNormal(mean, cov_mat)
        # Sample an action from the distribution and get its log prob
        if reparameterize:
            action = dist.rsample()
        else:
            action = dist.sample()
        # action = torch.flip(action, dims=(1,))
        log_prob = dist.log_prob(action)
        return action, log_prob  

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        chkpt = torch.load(self.checkpoint_file)
        self.load_state_dict(chkpt)













