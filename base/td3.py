import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import ReplayBuffer, soft_update

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
# Paper: https://arxiv.org/abs/1802.09477


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, max_action=1):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, action_dim)

        self.max_action = max_action

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))

    def get_action(self, state):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        action = self.forward(state)
        action = action.detach().cpu().numpy()[0]
        return action


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(Critic, self).__init__()

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1

    def get_q_values(self, state, action):
        state = torch.tensor(state, dtype=torch.float32).to(device)
        action = torch.tensor(action, dtype=torch.float32).to(device)
        q_value = self.Q1(state, action)
        q_value = q_value.detach().cpu().numpy()
        return q_value

class TD3(object):
    def __init__(
            self, state_dim, action_dim, summary_writer=None, max_action=1, gamma=0.99, batch_size=128,
            tau=0.005, policy_noise=0.2, expl_noise=0.1, noise_clip=0.5, policy_freq=2, buffer_size=10000,
            hidden_dim=256, critic_lr=3e-4, actor_lr=3e-4, actor_weight_decay=0., critic_weight_decay=0.
    ):
        self.actor = Actor(state_dim, action_dim, hidden_dim, max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr,
                                                weight_decay=actor_weight_decay)

        self.critic = Critic(state_dim, action_dim, hidden_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=critic_lr,
                                                 weight_decay=critic_weight_decay)

        self.replay_buffer = ReplayBuffer(state_dim, action_dim, buffer_size)
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.expl_noise = expl_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.summary_writer = summary_writer
        self.batch_size = batch_size

        self.action_dim = action_dim
        self.max_action = max_action
        self.t = 0

    def predict(self, state, with_noise=True):
        self.actor.eval()
        action = self.actor.get_action(state)
        if self.expl_noise and with_noise:
            action = (
                      action + np.random.normal(0, self.max_action * self.expl_noise, size=self.action_dim)
            ).clip(-self.max_action, self.max_action)
        self.actor.train()
        return action

    def update(self):

        # Sample replay buffer
        state, action, next_state, reward, not_done = self.replay_buffer.sample(self.batch_size)

        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (
                    torch.randn_like(action) * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip)

            next_action = (
                    self.actor_target(next_state) + noise
            ).clamp(-self.max_action, self.max_action)

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + not_done * self.gamma * target_Q

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Delayed policy updates
        if self.t % self.policy_freq == 0:

            # Compute actor loss
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            soft_update(self.critic_target, self.critic, self.tau)
            soft_update(self.actor_target, self.actor, self.tau)

        if self.summary_writer:

            grad_critic = torch.cat([p.grad.flatten() for p in self.critic.parameters()])
            self.summary_writer.add_scalar('loss/Critic_loss', critic_loss, self.t)
            self.summary_writer.add_scalar('extra/Critic_gradient_norm', grad_critic.norm(), self.t)

            try:
                grad_actor = torch.cat([p.grad.flatten() for p in self.actor.parameters()])
                self.summary_writer.add_scalar('loss/Actor_loss', actor_loss, self.t)
                self.summary_writer.add_scalar('extra/Actor_gradient_norm', grad_actor.norm(), self.t)
            except (NameError, AttributeError) as e:
                pass

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