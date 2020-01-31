import wolpertinger.knn_search as knn_search
import copy
from stable_baselines import DDPG
import tensorflow as tf

import gym
from gym.core import Env

import numpy as np

class DummyEnv(Env):
    def __init__(self, action_space, observation_space, reward_range=None):
        self.action_space = action_space
        self.observation_space = observation_space
        self.reward_range = reward_range

class WolpertingerAgent(DDPG):
    def __init__(self, policy, env, k_ratio=0.1, embeddings=None, **kwargs):

        if isinstance(env.action_space, gym.spaces.Discrete):
            n = env.action_space.n
            env_ = copy.deepcopy(env)
            env_.action_space = gym.spaces.Box(np.array([-1.] * n), np.array([1.] * n))
        elif isinstance(env.action_space, gym.spaces.MultiDiscrete) and len(env.action_space.nvec.shape) == 1:
            n = env.action_space.nvec[0]
            # in current SB version action space must be symmetric
            dummy_env = DummyEnv(action_space=gym.spaces.Box(np.array([-1.] * n), np.array([1.] * n)),
                                 observation_space=gym.spaces.Box(np.array([0.] * n), np.array([1.] * n)))
        else:
            raise Exception("Action space must be Discrete or one-dimensional MultiDiscrete")

        super(WolpertingerAgent, self).__init__(policy, dummy_env, **kwargs)

        self.knn_search = knn_search.KNNSearch(dummy_env.action_space, embeddings)
        self.k = max(1, int(n * k_ratio))

    def _policy(self, obs, apply_noise=True, compute_q=True):
        """
        Get the actions and critic output, from a given observation

        :param obs: ([float] or [int]) the observation
        :param apply_noise: (bool) enable the noise
        :param compute_q: (bool) compute the critic output
        :return: ([float], float) the action and critic value
        """
        obs = np.array(obs).reshape((-1,) + self.observation_space.shape)
        feed_dict = {self.obs_train: obs}
        if self.param_noise is not None and apply_noise:
            actor_tf = self.perturbed_actor_tf
            feed_dict[self.obs_noise] = obs
        else:
            actor_tf = self.actor_tf

        proto_action = self.sess.run(actor_tf, feed_dict=feed_dict)
        proto_action = proto_action.flatten()
        if self.action_noise is not None and apply_noise:
            noise = self.action_noise()
            assert noise.shape == proto_action.shape
            proto_action += noise
        proto_action = np.clip(proto_action, -1, 1)

        actions = self.knn_search.search_point(proto_action, self.k)[0]  # the nearest neighbour actions
        states = np.tile(obs, [len(actions), 1])  # make all the state-action pairs for the critic

        feed_dict = {self.obs_train: states, self.actions: actions}
        q_values = self.sess.run(self.critic_tf, feed_dict=feed_dict)

        max_index = np.argmax(q_values)  # find the index of the pair with the maximum value
        action, q_value = actions[max_index], q_values[max_index]
        action = (action + 1) / 2
        return action, q_value

    def _train_step(self, step, writer, log=False):
        with self.sess.as_default(), self.graph.as_default():
            return super()._train_step(step, writer, log)

    def get_sess(self):
        return self.sess

    def get_graph(self):
        return self.graph
