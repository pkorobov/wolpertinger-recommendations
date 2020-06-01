from .ddpg import DDPG
import matrix_env
import config
import gym
from gym.core import Env
import plots
import numpy as np
import faiss
import torch
import numpy as np
from .utils import ReplayBuffer, soft_update
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Wolpertinger(DDPG):
    def __init__(self, state_dim, action_dim, env, batch_size=128, gamma=0.99,
                 k_ratio=0.1, training_starts=100, eps=1e-2, embeddings=None, **kwargs):

        super(Wolpertinger, self).__init__(state_dim, action_dim,
                                           batch_size=batch_size, gamma=gamma,
                                           min_action=embeddings.min(axis=0),
                                           max_action=embeddings.max(axis=0),
                                           **kwargs)

        self.training_starts = training_starts
        self.eps = eps
        self.episode = None
        self.last_proto = None

        n, d = embeddings.shape
        self.embeddings = embeddings

        self.index = faiss.IndexFlatL2(d)

        if torch.cuda.is_available():
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)

        self.index.add(embeddings.astype(np.float32))
        self.k = max(1, int(n * k_ratio))

        D, _ = self.index.search(self.embeddings, 2)
        self.nn_distances = D[:, 1]

    def predict(self, state):

        proto_action = super().predict(state)
        proto_action = proto_action.clip(-1, 1)

        D, I = self.index.search(proto_action[np.newaxis, :].astype(np.float32), self.k)
        actions = self.embeddings[I[0]]
        states = np.tile(state, [len(actions), 1])  # make all the state-action pairs for the critic
        q_values = self.critic.get_q_values(states, actions)
        max_index = np.argmax(q_values)  # find the index of the pair with the maximum value
        action, index = actions[max_index], I[0][max_index]
        nearest_action = self.embeddings[I[0, 0]]
        return action, index, proto_action, nearest_action, self.nn_distances[index]

    def proto_action(self, state, with_noise=False):
        return super().predict(state, with_noise)

    def compute_q_values(self, state_num=0, target=False):
        s = np.tile(self.embeddings[state_num], (self.embeddings.shape[0], 1))
        a = self.embeddings

        if not target:
            return self.critic.get_q_values(s, a)
        return self.critic_target.get_q_values(s, a)

    def target_action(self, next_state):
        next_action = super().target_action(next_state)
        I = self.index.search(next_action.cpu().detach().numpy(), self.k)[1]
        proto_action_neighbours = self.embeddings[I]
        proto_action_neighbours = torch.from_numpy(proto_action_neighbours).to(device)
        next_state_tiled = next_state.unsqueeze(1).repeat(1, self.k, 1)

        q_values = self.critic_target(next_state_tiled, proto_action_neighbours).squeeze()
        next_action = torch.from_numpy(self.embeddings[q_values.argmax(dim=-1).cpu().detach()]).to(device)
        return next_action

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        state, action, proto_action, nearest_action, nn_distance, next_state, reward, done = self.replay_buffer.sample(self.batch_size)

        current_q = self.critic(state, action)
        # current_q = self.critic(state, action + torch.randn_like(action) * nn_distance / 2)
        target_q = self.critic_target(next_state, self.target_action(next_state))
        target_q = reward + ((1.0 - done) * self.gamma * target_q).detach()
        critic_loss = F.mse_loss(current_q, target_q) + \
                      0.1 * F.mse_loss(self.critic(state, proto_action), self.critic(state, nearest_action).detach())  # proto

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = self.critic(state, self.actor(state))
        actor_loss = -actor_loss.mean()

        if self.t % 5 == 0:
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
