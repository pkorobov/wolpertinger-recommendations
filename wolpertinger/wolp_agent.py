import wolpertinger.knn_search as knn_search
import copy
import torch
from base.ddpg import DDPG
from base.ddpg import Critic
from base.ddpg import Actor
import environment

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


        # old and ugly
        if isinstance(env.action_space, gym.spaces.Discrete):
            n = env.action_space.n
            env_ = copy.deepcopy(env)
            env_.action_space = gym.spaces.Box(np.array([0.] * n), np.array([1.] * n))
        elif isinstance(env.action_space, gym.spaces.MultiDiscrete) and len(env.action_space.nvec.shape) == 1:
            n = env.action_space.nvec[0]
            dummy_env = DummyEnv(action_space=gym.spaces.Box(np.array([0.] * n), np.array([1.] * n)),
                                 observation_space=gym.spaces.Box(np.array([0.] * n), np.array([1.] * n)))
        else:
            raise Exception("Action space must be Discrete or one-dimensional MultiDiscrete")

        self.knn_search = knn_search.KNNSearch(dummy_env.action_space, embeddings)
        self.k = max(1, int(n * k_ratio))

    def predict(self, state):

        proto_action = super().predict(state)
        proto_action = proto_action.clip(0, 1)

        actions = np.eye(self.action_dim)[np.lexsort((np.random.random(self.action_dim), proto_action))[-self.k:]]
        states = np.tile(state, [len(actions), 1])  # make all the state-action pairs for the critic
        q_values = self.critic.get_q_values(states, actions)
        max_index = np.argmax(q_values)  # find the index of the pair with the maximum value
        action, q_value = actions[max_index], q_values[max_index]
        return action

    def update(self):
        super().update()

    def compute_q_values(self, state_num=0, a=None, dim=None):
        if dim is None:
            dim = self.action_dim
        if a is None:
            a = np.eye(dim, self.action_dim)
        s = np.zeros((dim, self.action_dim))
        s[:, state_num] = 1
        q_vector = self.critic.get_q_values(s, a)
        return q_vector

    def compute_q_values_target(self, state_num=0, a=None, dim=None):
        if dim is None:
            dim = self.action_dim
        if a is None:
            a = np.eye(dim, self.action_dim)
        s = np.zeros((dim, self.action_dim))
        s[:, state_num] = 1
        q_vector = self.critic_target.get_q_values(s, a)
        return q_vector
