import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.normal import Normal
import numpy as np


class CriticNetwork(nn.Module):
    def __init__(self, 
                 beta, 
                 input_dims, 
                 n_actions, 
                 fc1_dims=128, 
                 fc2_dims=64, 
                 fc3_dims=32, 
                 name='critic', 
                 chkpt_dir='chkpt/') -> None:
        super(CriticNetwork, self).__init__()
        self.name = name
        self.beta = beta

        # layer dims
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.fc3_dims = fc3_dims
        self.n_actions = n_actions

        # Chkpt directory and filename
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, self.name+'_sac')

        # Model layers
        # self.fc1 = nn.Linear(self.input_dims[0]+self.n_actions, self.fc1_dims)
        # self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        # self.fc3 = nn.Linear(self.fc2_dims, self.fc3_dims)
        # self.q = nn.Linear(self.fc3_dims, 1)
        
        self.code_size = 95
        self.nav_size = 6
        
        self.reshape = nn.Sequential(
                        nn.Linear(2048, self.code_size),
                        nn.Tanh(),
                        nn.Linear(self.code_size, self.code_size),
                        nn.Tanh(),
                    )
        
        self.critic = nn.Sequential(
                        nn.Linear(self.code_size+self.nav_size+self.n_actions, 128),
                        nn.Tanh(),
                        nn.Linear(128, 64),
                        nn.Tanh(),
                        nn.Linear(64, 32),
                        nn.Tanh(),
                        nn.Linear(32, 1)
                    )

        # initialize the optimizer with LR
        self.opt = optim.Adam(self.parameters(), lr=beta)

        self.device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, obs, action):
        # Predict the Q-value given the state action pair
        obs_a = obs[:,:-self.nav_size]
        obs_n = obs[:,-self.nav_size:]
        rs_obs_a = self.reshape(obs_a)
        obs = torch.cat((rs_obs_a, obs_n), -1)
        
        input = torch.cat([obs, action], dim=-1)
        # x = self.fc1(input)
        # x = F.tanh(x)
        # x = self.fc2(x)
        # x = F.tanh(x)
        # x = self.fc3(x)
        # x = F.tanh(x)
        # q = self.q(x)
        q = self.critic(input)
        return q
    
    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        chkpt = torch.load(self.checkpoint_file)
        torch.load_state_dict(chkpt)
