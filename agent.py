from recsim.agent import AbstractEpisodicRecommenderAgent
from gym import spaces

import numpy as np

from wolpertinger.wolp_agent import *
from wolpertinger.ddpg.agent import DDPGAgent
from wolpertinger.util import data as util_data
from wolpertinger.util.timer import Timer


from recsim.simulator import recsim_gym, environment, runner_lib
from environment import *

class StaticAgent(AbstractEpisodicRecommenderAgent):

    def __init__(self, action_space, recommended_doc_id):
        super(StaticAgent, self).__init__(action_space)
        self.recommended_doc_id = recommended_doc_id

    def step(self, reward, observation):
        return [self.recommended_doc_id]

class WolpAgent(AbstractEpisodicRecommenderAgent):

    def __init__(self,
                 sess,
                 env,
                 observation_space,
                 action_space,
                 optimizer_name='',
                 eval_mode=False,
                 k_ratio=0.1,
                 episodes=2500,
                 max_actions=1000,
                 **kwargs):

        AbstractEpisodicRecommenderAgent.__init__(self, action_space)

        self._num_candidates = int(action_space.nvec[0])
        num_actions = self._num_candidates

        self._observation_space = env.observation_space
        self._action_space = spaces.Discrete(num_actions)
        self.k_nearest_neighbors = max(1, int(num_actions * k_ratio))

        self.data = util_data.Data()
        self.agent = WolpertingerAgent(env, max_actions=max_actions, k_ratio=self.k_nearest_neighbors)
        # self.data.set_experiment(experiment, agent.low.tolist(), agent.high.tolist(), episodes)
        self.agent.add_data_fetch(self.data)

        self.current_episode = {}

    def step(self, reward, observation):
        try:
            self.prev_obs
        except AttributeError:
            self.prev_obs = np.array([0] * self._observation_space.spaces['user'].n)
            self.prev_obs[-1] = 1

        user_space = self._observation_space.spaces['user']
        user_ohe = spaces.flatten(user_space, observation['user'])
        action = self.agent.act(user_ohe)

        self.current_episode = {
            'obs': self.prev_obs,
            'action': action,
            'reward': reward,
            'obs2': user_ohe,
            'done': 0,
        }

        self.agent.observe(self.current_episode)
        self.prev_obs = user_ohe

        return action
