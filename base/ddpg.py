import math
import random

import gym
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import itertools
import copy
from .utils import ReplayBuffer, soft_update

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


class GaussNoise:
    def __init__(self, sigma):
        super().__init__()

        self.sigma = sigma

    def get_action(self, action):
        noisy_action = np.random.normal(action, self.sigma)
        return noisy_action


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size, min_action=-1, max_action=1, init_w=3e-3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.head = nn.Linear(hidden_size, action_dim)
        nn.init.uniform_(self.head.weight, -init_w, init_w)
        nn.init.zeros_(self.head.bias)

        self.max_action = torch.tensor(max_action, dtype=torch.float32, device=device).detach()
        self.min_action = torch.tensor(min_action, dtype=torch.float32, device=device).detach()

    def forward(self, state):
        x = self.net(state)
        x = self.head(x)
        x = torch.tanh(x)
        return x * (self.max_action - self.min_action) / 2 + (self.max_action + self.min_action) / 2

    def get_action(self, state):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        action = self.forward(state)
        action = action.detach().cpu().numpy()[0]
        return action


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size, init_w=3e-3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )

        # it seems better to have tanh activations and no special initialization
        self.head = nn.Linear(hidden_size, 1)
        # nn.init.uniform_(self.head.weight, -init_w, init_w)
        # nn.init.zeros_(self.head.bias)

    def forward(self, state, action):
        x = torch.cat([state, action], -1)
        x = self.net(x)
        x = self.head(x)
        return x

    def get_q_values(self, state, action):
        state = torch.tensor(state, dtype=torch.float32).to(device)
        action = torch.tensor(action, dtype=torch.float32).to(device)
        q_value = self.forward(state, action)
        q_value = q_value.detach().cpu().numpy()
        return q_value


class DDPG:
    def __init__(self, state_dim, action_dim, summary_writer=None, expl_noise=0.1,
                 buffer_size=10000, hidden_dim=256, tau=1e-3, batch_size=128,
                 gamma=0.99, init_w_actor=3e-3, init_w_critic=3e-3, critic_lr=1e-3,
                 actor_lr=1e-4, actor_weight_decay=0., critic_weight_decay=0.,
                 max_action=1, min_action=-1
    ):

        self.actor = Actor(state_dim, action_dim, hidden_dim, min_action, max_action, init_w=init_w_actor).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = optim.Adam(self.actor.parameters(),
                                          lr=actor_lr,
                                          weight_decay=actor_weight_decay)

        self.critic = Critic(state_dim, action_dim, hidden_dim, init_w=init_w_critic).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = optim.Adam(self.critic.parameters(),
                                           lr=critic_lr,
                                           weight_decay=critic_weight_decay)

        self.action_dim = action_dim
        self.replay_buffer = ReplayBuffer(state_dim, action_dim, buffer_size)
        self.tau = tau
        self.expl_noise = expl_noise
        self.summary_writer = summary_writer
        self.batch_size = batch_size
        self.gamma = gamma
        self.max_action = max_action
        self.min_action = min_action
        self.t = 0

    def target_action(self, next_state):
        return self.actor_target(next_state)

    def predict(self, state, with_noise=True):
        self.actor.eval()
        action = self.actor.get_action(state)
        if self.expl_noise and with_noise:
            action = (
                      action + np.random.normal(0, (self.max_action - self.min_action) / 2 * self.expl_noise, size=self.action_dim)
            ).clip(-self.min_action, self.max_action)
        self.actor.train()
        return action

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        state, action, next_state, reward, done = self.replay_buffer.sample(self.batch_size)

        current_q = self.critic(state, action)
        target_q = self.critic_target(next_state, self.target_action(next_state))
        target_q = reward + ((1.0 - done) * self.gamma * target_q).detach()
        critic_loss = F.mse_loss(current_q, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = self.critic(state, self.actor(state))
        actor_loss = -actor_loss.mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        if self.summary_writer:

            grad_actor = torch.cat([p.grad.flatten() for p in self.actor.parameters()])
            grad_critic = torch.cat([p.grad.flatten() for p in self.critic.parameters()])

            self.summary_writer.add_scalar('loss/Actor_loss', actor_loss, self.t)
            self.summary_writer.add_scalar('loss/Critic_loss', critic_loss, self.t)
            self.summary_writer.add_scalar('extra/Actor_gradient_norm', grad_actor.norm(), self.t)
            self.summary_writer.add_scalar('extra/Critic_gradient_norm', grad_critic.norm(), self.t)

        soft_update(self.critic_target, self.critic, self.tau)
        soft_update(self.actor_target, self.actor, self.tau)

    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")

        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")

    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.actor_target = copy.deepcopy(self.actor)
