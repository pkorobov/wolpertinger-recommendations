import copy
from stable_baselines import SAC
import tensorflow as tf

import gym
from gym.core import Env

import numpy as np

class DummyEnv(Env):
    def __init__(self, action_space, observation_space, reward_range=None):
        self.action_space = action_space
        self.observation_space = observation_space
        self.reward_range = reward_range

class SoftWolpertingerAgent(SAC):
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

        super(SoftWolpertingerAgent, self).__init__(policy, dummy_env, **kwargs)

        self.knn_search = knn_search.KNNSearch(dummy_env.action_space, embeddings)
        self.k = max(1, int(n * k_ratio))

    def predict(self, observation, state=None, mask=None, deterministic=False):
        observation = np.array(observation)
        vectorized_env = self._is_vectorized_observation(observation, self.observation_space)

        observation = observation.reshape((-1,) + self.observation_space.shape)
        proto_action = self.policy_tf.step(observation, deterministic=deterministic)
        proto_action = proto_action.reshape((-1,) + self.action_space.shape)  # reshape to the correct action shape
        proto_action = proto_action * np.abs(self.action_space.low)  # scale the output for the prediction

        if not vectorized_env:
            proto_action = proto_action[0]

        actions = self.knn_search.search_point(proto_action, self.k)[0]  # the nearest neighbour actions
        states = np.tile(observation, [len(actions), 1])  # make all the state-action pairs for the critic

        qf1 = self.step_ops[4]
        feed_dict = {self.observations_ph: states, self.actions_ph: actions}
        q_values = self.sess.run(qf1, feed_dict=feed_dict)

        max_index = np.argmax(q_values)  # find the index of the pair with the maximum value
        action, q_value = actions[max_index], q_values[max_index]
        action = (action + 1) / 2
        return action, None

    def _train_step(self, step, writer, learning_rate=1e-3):
        with self.sess.as_default(), self.graph.as_default():
            return super()._train_step(step, writer, learning_rate)

    def get_sess(self):
        return self.sess

    def get_graph(self):
        return self.graph
