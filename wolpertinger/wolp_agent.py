import wolpertinger.knn_search as knn_search
import copy
import torch
from base.ddpg import DDPG
from base.ddpg import CriticNetwork
from base.ddpg import ActorNetwork

import gym
from gym.core import Env

import numpy as np

class DummyEnv(Env):
    def __init__(self, action_space, observation_space, reward_range=None):
        self.action_space = action_space
        self.observation_space = observation_space
        self.reward_range = reward_range

class WolpertingerAgent(DDPG):
    def __init__(self, critic_constructor, actor_constructor, state_dim, action_dim, env,
                 batch_size=128, gamma=0.99, min_value = -np.inf, max_value = np.inf, tau = 1e-2,
                 k_ratio=0.1, embeddings=None, **kwargs):

        super(WolpertingerAgent, self).__init__(critic_constructor,
                                                actor_constructor,
                                                state_dim, action_dim,
                                                **kwargs)
        self.batch_size = batch_size
        self.gamma = gamma
        self.min_value = min_value
        self.max_value = max_value
        self.tau = tau

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
        proto_action = np.clip(proto_action, -1, 1)

        actions = self.knn_search.search_point(proto_action, self.k)[0]  # the nearest neighbour actions
        states = np.tile(state, [len(actions), 1])  # make all the state-action pairs for the critic
        q_values = self.critic_net.get_q_values(states, actions)

        max_index = np.argmax(q_values)  # find the index of the pair with the maximum value
        action, q_value = actions[max_index], q_values[max_index]
        return action

    def update(self):
        return super().update(self.batch_size, self.gamma,
                              self.min_value, self.max_value,
                              self.tau)