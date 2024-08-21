import os
import numpy as np

import torch
import torch.nn as nn
from algos.ppo.ppo import ActorCritic
from parameters import  *
from encoder.zoo import EncoderZoo
from utils.transform import TransformObservation
from tqdm import tqdm


device = torch.device("cpu")

class Buffer:
    def __init__(self):
         # Batch data
        self.observation = []  
        self.observation_nav = []  
        self.actions = []         
        self.log_probs = []     
        self.rewards = []         
        self.dones = []

    def clear(self):
        del self.observation[:]
        del self.observation_nav[:]    
        del self.actions[:]        
        del self.log_probs[:]      
        del self.rewards[:]
        del self.dones[:]

class PPOAgent(object):
    def __init__(self, town, action_std_init=0.4):
        
        #self.env = env
        self.obs_dim = 2048
        self.action_dim = 2
        self.clip = POLICY_CLIP
        self.gamma = GAMMA
        self.n_updates_per_iteration = 8
        self.lr = PPO_LEARNING_RATE
        self.action_std = action_std_init
        self.memory = Buffer()
        self.town = town

        self.tfm = TransformObservation(device)
        self.encoder_zoo = EncoderZoo(CODE_SIZE)
        self.en_model = self.encoder_zoo.get_model(ENCODER_MODEL)

        self.checkpoint_file_no = 0
        
        self.policy = ActorCritic(self.obs_dim, self.action_dim, self.action_std)
        self.optimizer = torch.optim.Adam([
                        {'params': self.policy.actor.parameters(), 'lr': self.lr},
                        {'params': self.policy.critic.parameters(), 'lr': self.lr}])

        self.old_policy = ActorCritic(self.obs_dim, self.action_dim, self.action_std)
        self.old_policy.load_state_dict(self.policy.state_dict())
        self.MseLoss = nn.MSELoss()


    def get_action(self, obs, train):
        obs = self.tfm.transform(obs)
        with torch.no_grad():
            if isinstance(obs, np.ndarray):
                obs = torch.tensor(obs, dtype=torch.float)
            action, logprob = self.old_policy.get_action_and_log_prob(obs.to(device))
        if train:
            self.memory.observation.append(obs.to(device))
            self.memory.actions.append(action)
            self.memory.log_probs.append(logprob)
        return action.detach().cpu().numpy().flatten()
    
    def set_action_std(self, new_action_std):
        self.action_std = new_action_std
        self.policy.set_action_std(new_action_std)
        self.old_policy.set_action_std(new_action_std)

    
    def decay_action_std(self, action_std_decay_rate, min_action_std):
        self.action_std = self.action_std - action_std_decay_rate
        if (self.action_std <= min_action_std):
            self.action_std = min_action_std
        self.set_action_std(self.action_std)
        return self.action_std


    def train(self):
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.memory.rewards), reversed(self.memory.dones)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        
        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)
        # convert list to tensor
        old_states = torch.squeeze(torch.stack(self.memory.observation, dim=0)).detach().to(device)
        old_actions = torch.squeeze(torch.stack(self.memory.actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(self.memory.log_probs, dim=0)).detach().to(device)
        # Optimize policy for K epochs
        for _ in tqdm(range(self.n_updates_per_iteration)):

            # Evaluating old actions and values
            logprobs, values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # match values tensor dimensions with rewards tensor
            values = torch.squeeze(values)
            
            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            advantages = rewards - values.detach()   
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.clip, 1+self.clip) * advantages

            # final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(values, rewards) - 0.01*dist_entropy
            
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
        self.old_policy.load_state_dict(self.policy.state_dict())
        self.memory.clear()
    
    def save(self):
        self.checkpoint_file_no = len(next(os.walk(PPO_CHECKPOINT_DIR+self.town))[2])
        checkpoint_file = PPO_CHECKPOINT_DIR+self.town+"/ppo_policy_" + str(self.checkpoint_file_no)+"_.pth"
        torch.save(self.old_policy.state_dict(), checkpoint_file)

    def chkpt_save(self):
        self.checkpoint_file_no = len(next(os.walk(PPO_CHECKPOINT_DIR+self.town))[2])
        if self.checkpoint_file_no !=0:
            self.checkpoint_file_no -=1
        checkpoint_file = PPO_CHECKPOINT_DIR+self.town+"/ppo_policy_" + str(self.checkpoint_file_no)+"_.pth"
        torch.save(self.old_policy.state_dict(), checkpoint_file)
   
    def load(self):
        if CHKPT:
            self.checkpoint_file_no = CHKPT - 1
        else:
            self.checkpoint_file_no = len(next(os.walk(PPO_CHECKPOINT_DIR+self.town))[2]) - 2
        checkpoint_file = PPO_CHECKPOINT_DIR+self.town+"/ppo_policy_" + str(self.checkpoint_file_no)+"_.pth"
        self.old_policy.load_state_dict(torch.load(checkpoint_file))
        self.policy.load_state_dict(torch.load(checkpoint_file))