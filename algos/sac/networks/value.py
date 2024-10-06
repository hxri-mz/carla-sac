import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.normal import Normal
import numpy as np

class ValueNetwork(nn.Module):
    def __init__(self, 
                 beta, 
                 input_dims, 
                 fc1_dims=128, 
                 fc2_dims=64, 
                 fc3_dims=32, 
                 name='value', 
                 chkpt_dir='chkpt/') -> None:
        super(ValueNetwork, self).__init__()
        self.beta = beta

        self.name = name
        self.beta = beta

        # layer dims
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.fc3_dims = fc3_dims

        # Chkpt directory and filename
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, self.name+'_sac')
        
        self.code_size = 95
        self.nav_size = 6
        
        # self.reshape = nn.Sequential(
        #                 nn.Linear(2048, self.code_size),
        #                 nn.Tanh(),
        #                 nn.Linear(self.code_size, self.code_size),
        #                 nn.Tanh(),
        #             )
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
        
        self.value = nn.Sequential(
                        nn.Linear(self.code_size+self.nav_size, 128),
                        nn.Tanh(),
                        nn.Linear(128, 64),
                        nn.Tanh(),
                        nn.Linear(64, 32),
                        nn.Tanh(),
                        nn.Linear(32, 1),
                    )

        # initialize the optimizer with LR
        self.opt = optim.Adam(self.parameters(), lr=beta)

        self.device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)


    def forward(self, obs):
        obs_a = obs[:,:-self.nav_size]
        obs_n = obs[:,-self.nav_size:]
        rs_obs_a = self.reshape(obs_a)
        obsn = torch.cat((rs_obs_a, obs_n), -1)

        v = self.value(obsn)
        return v
    
    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        chkpt = torch.load(self.checkpoint_file)
        self.load_state_dict(chkpt)
