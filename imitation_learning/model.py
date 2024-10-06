import torch.nn as nn
import torch
import sys

from torch.distributions import MultivariateNormal
from torch.optim import Adam

class ImitationModel(nn.Module):
    def __init__(self, obs_dim, code_size, nav_size, action_dim, device) -> None:
        super(ImitationModel, self).__init__()
        self.action_dim = action_dim 
        self.obs_dim = obs_dim
        self.code_size = code_size 
        self.nav_size = nav_size   
        self.device = device
        self.cov_var = torch.full((self.action_dim,), 1e-8)
        self.cov_mat = torch.diag(self.cov_var).unsqueeze(dim=0).to(self.device)
        

        self.cmd_process = nn.Sequential(
                        nn.Linear(3, 6),
                        nn.Tanh(),
                        nn.Linear(6, 8),
                        nn.Tanh()
                    )
        
        self.hist_process = nn.Sequential(
                        nn.Linear(3, 6),
                        nn.Tanh(),
                        nn.Linear(6, 8),
                        nn.Tanh()
                    )
        
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
                        nn.Linear(self.code_size+16, 128),
                        nn.Tanh(),
                        nn.Linear(128, 64),
                        nn.Tanh(),
                        nn.Linear(64, 32),
                        nn.Tanh(),
                        nn.Linear(32, self.action_dim),
                        nn.Tanh()
                    )
        
        self.opt = Adam(self.parameters(), lr=1e-4)
        self.to(self.device)
    
    def forward(self, obs, nav):
        # obs = torch.tensor(obs, dtype=torch.float)
        # nav = torch.tensor(nav, dtype=torch.float)
        n_proc = torch.cat((self.cmd_process(nav[:, -3:]), self.hist_process(nav[:, :3])), -1)
        obs_n = self.reshape(obs)
        obs_full = torch.cat((obs_n, n_proc), -1)
        action = self.actor(obs_full.to(torch.float32))
        # dist = MultivariateNormal(mean, self.cov_mat)
        # action = dist.rsample()
        
        return action