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
                 env,
                 sess,
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

        self._action_space = spaces.Discrete(num_actions)
        self.k_nearest_neighbors = max(1, int(num_actions * k_ratio))

        self.data = util_data.Data()
        self.agent = WolpertingerAgent(env, max_actions=max_actions, k_ratio=self.k_nearest_neighbors)
        # self.data.set_experiment(experiment, agent.low.tolist(), agent.high.tolist(), episodes)
        self.agent.add_data_fetch(self.data)

        self.current_episode = {}

    # def begin_episode(self, observation=None):
    #     pass

    def step(self, reward, observation):
        if self.current_episode is not None:
            self.current_episode["reward"] = reward
            self.agent.observe(self.current_episode)

        action = self.agent.act(observation)

        self.current_episode.clear()
        self.current_episode = {
            'obs': observation,
            'action': action,
            'reward': reward,
            'obs2': observation,
            # 'done': done,s
            # 't': t
        }

        return action
