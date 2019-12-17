from recsim.agent import AbstractEpisodicRecommenderAgent

from environment import *
from wolpertinger.wolp_agent import *

from stable_baselines.ddpg import MlpPolicy

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
        self.agent = WolpertingerAgent(MlpPolicy, env, k_ratio=k_ratio)

        self.t = 0
        self.current_episode = {}

    def begin_episode(self, observation=None):
        self.agent._setup_learn(0)
        state = self._extract_state(observation)
        return self._act(state)

    def step(self, reward, observation):
        state = self._extract_state(observation)
        self._observe(state, reward, 0)
        self.t += 1
        return self._act(state)

    def end_episode(self, reward, observation=None):
        state = self._extract_state(observation)
        self._observe(state, reward, 1)

    def _act(self, state):
        action, q_value = self.agent._policy(state)
        self.current_episode = {
            "obs": state,
            "action": action,
        }
        return np.where(action)[0]

    def _observe(self, next_state, reward, done):
        if not self.current_episode:
            raise ValueError("Current episode is expected to be non-empty")

        self.current_episode.update({
            "next_obs": next_state,
            "reward": reward,
            "done": done
        })

        self.agent._store_transition(**self.current_episode)
        if self.agent.replay_buffer.can_sample(self.agent.batch_size):
            critic_loss, actor_loss = self.agent._train_step(self.t, None)#, log=t_train == 0)
        self.current_episode = {}

    def _extract_state(self, observation):
        user_space = self._observation_space.spaces['user']
        return spaces.flatten(user_space, observation['user'])
