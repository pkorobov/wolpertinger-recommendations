from recsim.agent import AbstractEpisodicRecommenderAgent

from environment import *
from wolpertinger.util import data as util_data
from wolpertinger.wolp_agent import *


class StaticAgent(AbstractEpisodicRecommenderAgent):

    def __init__(self, action_space, recommended_doc_id):
        super(StaticAgent, self).__init__(action_space)
        self.recommended_doc_id = recommended_doc_id

    def step(self, reward, observation):
        return [self.recommended_doc_id]


class WolpAgent(AbstractEpisodicRecommenderAgent):

    def __init__(self, env, action_space, k_ratio=0.1, max_actions=1000):
        AbstractEpisodicRecommenderAgent.__init__(self, action_space)

        self._observation_space = env.observation_space

        num_actions = int(action_space.nvec[0])
        self.agent = WolpertingerAgent(env, max_actions=max_actions, k_ratio=k_ratio)
        self.agent.add_data_fetch(util_data.Data())

        self.t = 0
        self.current_episode = {}

    def begin_episode(self, observation=None):
        self.t = 0
        state = self._extract_state(observation)
        return self._act(state)

    def step(self, reward, observation):
        state = self._extract_state(observation)
        self._observe(state, reward, 0)
        return self._act(state)

    def end_episode(self, reward, observation=None):
        state = self._extract_state(observation)
        self._observe(state, reward, 1)

    def _act(self, state):
        action = self.agent.act(state)
        self.current_episode = {
            "obs": state,
            "action": action,
            "t": self.t
        }
        self.t += 1
        return np.where(action)[0]

    def _observe(self, next_state, reward, done):
        if not self.current_episode:
            raise ValueError("Current episode is expected to be non-empty")

        self.current_episode.update({
            "obs2": next_state,
            "reward": reward,
            "done": done
        })

        self.agent.observe(self.current_episode)
        self.current_episode = {}

    def _extract_state(self, observation):
        user_space = self._observation_space.spaces['user']
        return spaces.flatten(user_space, observation['user'])
