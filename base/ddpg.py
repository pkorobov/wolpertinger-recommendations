import math
import random

import gym
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal

use_cuda = torch.cuda.is_available()
device   = torch.device("cuda" if use_cuda else "cpu")

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

class GaussNoise:
    """
    For continuous environments only.
    Adds spherical Gaussian noise to the action produced by actor.
    """

    def __init__(self, sigma):
        super().__init__()

        self.sigma = sigma

    def get_action(self, action):
        noisy_action = np.random.normal(action, self.sigma)
        return noisy_action

class NormalizedActions(gym.ActionWrapper):

    def action(self, action):
        low_bound = self.action_space.low
        upper_bound = self.action_space.high

        action = low_bound + (action + 1.0) * 0.5 * (upper_bound - low_bound)
        action = np.clip(action, low_bound, upper_bound)

        return action

    def reverse_action(self, action):
        low_bound = self.action_space.low
        upper_bound = self.action_space.high

        action = 2 * (action - low_bound) / (upper_bound - low_bound) - 1
        action = np.clip(action, low_bound, upper_bound)
        return action

class CriticNetwork(nn.Module):
    def __init__(
            self,
            num_inputs,
            num_actions,
            hidden_size,
            init_w=3e-3
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_inputs + num_actions, hidden_size),
            nn.ReLU(),
            # nn.Linear(hidden_size, hidden_size),
            # nn.ReLU(),
        )
        self.head = nn.Linear(hidden_size, 1)

        self.head.weight.data.uniform_(-init_w, init_w)
        self.head.bias.data.uniform_(-init_w, init_w)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = self.net(x)
        x = self.head(x)
        return x

    def get_q_values(self, state, action):
        state = torch.tensor(state, dtype=torch.float32).to(device)
        action = torch.tensor(action, dtype=torch.float32).to(device)
        q_value = self.forward(state, action)
        q_value = q_value.detach().cpu().numpy()
        return q_value

class ActorNetwork(nn.Module):
    def __init__(
            self,
            num_inputs,
            num_actions,
            hidden_size,
            init_w=3e-3
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            # nn.Linear(hidden_size, hidden_size),
            # nn.ReLU(),
        )
        self.head = nn.Linear(hidden_size, num_actions)
        self.head.weight.data.uniform_(-init_w, init_w)
        self.head.bias.data.uniform_(-init_w, init_w)

    def forward(self, state):
        x = state
        x = self.net(x)
        x = self.head(x)
        return x

    def get_action(self, state):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        action = self.forward(state)
        action = action.detach().cpu().numpy()[0]
        return action

class DDPG:
    def __init__(self, critic_constructor, actor_constructor, state_dim,
                 action_dim, noise=None, buffer_size=1000, hidden_dim=16, critic_criterion=nn.MSELoss(),
                 critic_lr=1e-3, actor_lr = 1e-4, **kwargs):

        self.critic_net = critic_constructor(state_dim, action_dim, hidden_dim).to(device)
        self.actor_net = actor_constructor(state_dim, action_dim, hidden_dim).to(device)
        self.target_critic_net = critic_constructor(state_dim, action_dim, hidden_dim).to(device)
        self.target_actor_net = actor_constructor(state_dim, action_dim, hidden_dim).to(device)
        self.replay_buffer = ReplayBuffer(buffer_size)
        self.critic_criterion = critic_criterion

        self.critic_optimizer = optim.Adam(self.critic_net.parameters(), lr=critic_lr)
        self.actor_optimizer = optim.Adam(self.actor_net.parameters(), lr=actor_lr)
        self.noise = noise

    def predict(self, state):
        action = self.actor_net.get_action(state)
        if self.noise:
            action = self.noise.get_action(action)
        return action

    def update(self,
               batch_size,
               gamma=0.99,
               min_value=-np.inf,
               max_value=np.inf,
               soft_tau=1e-2,
    ):

        state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)

        state = torch.tensor(state, dtype=torch.float32).to(device)
        next_state = torch.tensor(next_state, dtype=torch.float32).to(device)
        action = torch.tensor(action, dtype=torch.float32).to(device)
        reward = torch.tensor(reward, dtype=torch.float32).unsqueeze(1).to(device)
        done = torch.tensor(np.float32(done)).unsqueeze(1).to(device)

        actor_loss = self.critic_net(state, self.actor_net(state))
        actor_loss = -actor_loss.mean()

        next_action = self.actor_net(next_state)
        target_value = self.target_critic_net(next_state, next_action.detach())
        expected_value = reward + (1.0 - done) * gamma * target_value
        expected_value = torch.clamp(expected_value, min_value, max_value)

        q_value = self.critic_net(state, action)
        critic_loss = self.critic_criterion(q_value, expected_value.detach())

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        for target_param, param in zip(self.target_critic_net.parameters(), self.critic_net.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - soft_tau) + param.data * soft_tau
            )

        for target_param, param in zip(self.target_actor_net.parameters(), self.actor_net.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - soft_tau) + param.data * soft_tau
            )