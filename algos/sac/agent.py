import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils.replay_memory import ReplayBuffer
from .networks import ActorNetwork, CriticNetwork, ValueNetwork
from .parameters import *

class SACAgent():
    def __init__(self,
                 alpha=ALPHA,
                 beta=BETA,
                 input_dims=INPUT_DIMS,
                 env=None,
                 gamma=GAMMA,
                 tau=TAU,
                 n_actions=N_ACTIONS,
                 max_size=MAX_SIZE,
                 layer1_size=LAYER1_SIZE,
                 layer2_size=LAYER2_SIZE,
                 layer3_size=LAYER3_SIZE,
                 batch_size=BATCH_SIZE,
                 reward_scale=REWARD_SCALE,
                 action_std_init=0.4) -> None:
        self.gamma = gamma
        self.tau = tau
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.n_actions = n_actions
        self.action_std = action_std_init
        self.num_epochs = 8

        # define the networks
        # TODO: create a max_action value in carla env
        self.actor = ActorNetwork(alpha=alpha, fc1_dims=layer1_size, fc2_dims=layer2_size, fc3_dims=layer3_size, input_dims=input_dims, n_actions=n_actions, action_std_init=self.action_std)

        # Critic networks
        self.critic_a = CriticNetwork(beta=beta, fc1_dims=layer1_size, fc2_dims=layer2_size, fc3_dims=layer3_size, input_dims=input_dims, n_actions=n_actions, name='critic_a')
        self.critic_b = CriticNetwork(beta=beta, fc1_dims=layer1_size, fc2_dims=layer2_size, fc3_dims=layer3_size, input_dims=input_dims, n_actions=n_actions, name='critic_b')

        # Value networks
        self.value = ValueNetwork(beta=beta, fc1_dims=layer1_size, fc2_dims=layer2_size, fc3_dims=layer3_size, input_dims=input_dims, name='value')
        self.value_target = ValueNetwork(beta=beta, fc1_dims=layer1_size, fc2_dims=layer2_size, fc3_dims=layer3_size, input_dims=input_dims, name='value_target')

        self.scale_factor = reward_scale
        self.update_network_parameters(tau=1)

    def set_action_std(self, new_action_std):
        self.action_std = new_action_std
        self.actor.set_action_std(new_action_std)

    def action_selection(self, observation):
        # import pdb; pdb.set_trace()
        # state = torch.tensor([obs], dtype=torch.float32).to(self.actor.device)
        state = observation.to(self.actor.device)
        actions, _ = self.actor.sample_normal(state, reparameterize=False)
        return actions.cpu().detach().numpy()[0]
    
    def memorize(self, state, action, reward, n_state, done):
        self.memory.store_transition(state, action, reward, n_state, done)

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        # Update value parameters
        target_params = self.value_target.named_parameters()
        value_params = self.value.named_parameters()

        target_state_dict = dict(target_params)
        value_state_dict = dict(value_params)

        # soft copy parameters
        for name in value_state_dict:
            value_state_dict[name] = tau*value_state_dict[name].clone() + (1-tau)*target_state_dict[name].clone()

        self.value_target.load_state_dict(value_state_dict)

    def learn(self):
        if self.memory.mem_counter < self.batch_size:
            print(self.memory.mem_counter)
            return
        else:
            # state, action, reward, n_state, done = self.memory.sample_buffer(self.batch_size)
            state, action, reward, n_state, done = self.memory.sample_buffer(self.batch_size)

            # put everything to cuda as tensors
            state = torch.tensor(state, dtype=torch.float32).to(self.actor.device)
            n_state = torch.tensor(n_state, dtype=torch.float32).to(self.actor.device)
            action = torch.tensor(action, dtype=torch.float32).to(self.actor.device)
            reward = torch.tensor(reward, dtype=torch.float32).to(self.actor.device)
            done = torch.tensor(done).to(self.actor.device)

            value = self.value(state).view(-1)
            target = self.value_target(n_state).view(-1)
            target[done] = 0.0

            # actions from new policy
            actions, log_probas = self.actor.sample_normal(state, reparameterize=True)
            log_probas = log_probas.view(-1)
            q_a_new_policy = self.critic_a(state, actions)
            q_b_new_policy = self.critic_b(state, actions)
            critic_value = torch.min(q_a_new_policy, q_b_new_policy) # take min of a and b networks to mitigate over estimation bias
            critic_value = critic_value.view(-1)

            # Backpropogate value loss
            self.value.opt.zero_grad()
            value_target = critic_value - log_probas
            value_loss = 0.5*F.mse_loss(value, value_target)
            value_loss.backward(retain_graph=True) # retain graph because there is coupling between various networks in SAC and we need to keep track of loss
            self.value.opt.step()

            # Backpropogate actor loss
            actions, log_probas = self.actor.sample_normal(state, reparameterize=True)
            log_probas = log_probas.view(-1)
            q_a_new_policy = self.critic_a(state, actions)
            q_b_new_policy = self.critic_b(state, actions)
            critic_value = torch.min(q_a_new_policy, q_b_new_policy)
            critic_value = critic_value.view(-1)
            actor_loss = log_probas - critic_value
            actor_loss = torch.mean(actor_loss)
            self.actor.opt.zero_grad()
            actor_loss.backward(retain_graph=True)
            self.actor.opt.step()

            # Backpropogate critic loss
            self.critic_a.opt.zero_grad()
            self.critic_b.opt.zero_grad()
            q_ = self.scale_factor*reward + self.gamma*target
            q_a_old_policy = self.critic_a.forward(state, action=action).view(-1)
            q_b_old_policy = self.critic_b.forward(state, action=action).view(-1)
            critic_a_loss = 0.5*F.mse_loss(q_a_old_policy, q_)
            critic_b_loss = 0.5*F.mse_loss(q_b_old_policy, q_)
            critic_loss = critic_a_loss + critic_b_loss
            critic_loss.backward()
            self.critic_a.opt.step()
            self.critic_b.opt.step()

            self.update_network_parameters()

            print(f"Training Metrics - Actor Loss: {actor_loss} | Critic Loss: {critic_loss} | Value Loss: {value_loss}")


    def save_models(self):
        print("--------- Saving models -----------")
        self.actor.save_checkpoint()
        self.value.save_checkpoint()
        self.value_target.save_checkpoint()
        self.critic_a.save_checkpoint()
        self.critic_b.save_checkpoint()

    def load_models(self):
        print("--------- Loading models -----------")
        self.actor.load_checkpoint()
        self.value.load_checkpoint()
        self.value_target.load_checkpoint()
        self.critic_a.load_checkpoint()
        self.critic_b.load_checkpoint()

    

