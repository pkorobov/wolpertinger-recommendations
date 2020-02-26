import copy
from base.ddpg import DDPG
import environment
import faiss

import gym
from gym.core import Env

import numpy as np

class DummyEnv(Env):
    def __init__(self, action_space, observation_space, reward_range=None):
        self.action_space = action_space
        self.observation_space = observation_space
        self.reward_range = reward_range

class WolpertingerAgent(DDPG):
    def __init__(self, state_dim, action_dim, env,
                 batch_size=128, gamma=0.99, min_value=-np.inf, max_value=np.inf,
                 k_ratio=0.1, embeddings=None, **kwargs):

        super(WolpertingerAgent, self).__init__(state_dim, action_dim,
                                                batch_size=batch_size, gamma=gamma,
                                                min_value=min_value, max_value=max_value,
                                                **kwargs)

        n, d = embeddings.shape
        self.embeddings = embeddings
        self.index = faiss.IndexFlatL2(d)
        self.index.add(embeddings.astype(np.float32))

        self.k = max(1, int(n * k_ratio))

    def predict(self, state):

        proto_action = super().predict(state)
        proto_action = proto_action.clip(0, 1)

        _, I = self.index.search(proto_action[np.newaxis, :].astype(np.float32), self.k)
        actions = self.embeddings[I[0]]
        states = np.tile(state, [len(actions), 1])  # make all the state-action pairs for the critic
        q_values = self.critic.get_q_values(states, actions)
        max_index = np.argmax(q_values)  # find the index of the pair with the maximum value
        action, q_value = actions[max_index], q_values[max_index]
        return action

    def update(self):
        super().update()

    def compute_q_values(self, state_num=0, a=None, dim=None):
        if dim is None:
            dim = self.embeddings.shape[0]
        if a is None:
            a = self.embeddings
        s = self.embeddings[state_num]
        s = np.tile(s, [dim, 1])
        q_vector = self.critic.get_q_values(s, a)
        return q_vector

    def compute_q_values_target(self, state_num=0, a=None, dim=None):
        if dim is None:
            dim = self.embeddings.shape[0]
        if a is None:
            a = self.embeddings
        s = self.embeddings[state_num]
        s = np.tile(s, [dim, 1])
        q_vector = self.critic_target.get_q_values(s, a)
        return q_vector
