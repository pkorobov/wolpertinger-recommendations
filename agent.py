from recsim.agent import AbstractEpisodicRecommenderAgent
from base.wolpertinger import *
from base.ddpg import DDPG
from gym import spaces
import config as c

class OptimalAgent(AbstractEpisodicRecommenderAgent):

    def __init__(self, env):
        super(OptimalAgent, self).__init__(env.action_space)
        self._observation_space = env.observation_space

    def step(self, reward, observation):
        doc_id = observation['user'][0]
        return [c.W[doc_id, :].argmax()]


class WolpertingerRecSim(AbstractEpisodicRecommenderAgent):

    def __init__(self, env, state_dim, action_dim, backend=DDPG,
                 k_ratio=0.1, eval_mode=False, **kwargs):
        AbstractEpisodicRecommenderAgent.__init__(self, env.action_space)

        self._observation_space = env.observation_space

        self.agent = createWolpertinger(backend)(state_dim, action_dim,
                                                 env, k_ratio=k_ratio, **kwargs)
        self.agent.t = 0
        self.current_episode = {}
        self.eval_mode = eval_mode

    def begin_episode(self, observation=None):
        state = self._extract_state(observation)
        return self._act(state)

    def step(self, reward, observation):
        state = self._extract_state(observation)
        self._observe(state, reward, 0)
        self.agent.t += 1
        return self._act(state)

    def end_episode(self, reward, observation=None):
        state = self._extract_state(observation)
        self._observe(state, reward, 1)

    def _act(self, state):
        if np.random.rand() < self.agent.eps or \
           self.agent.t < self.agent.training_starts:
            index = np.random.randint(c.DOC_NUM)
            action = self.agent.embeddings[index]
        else:
            action, index = self.agent.predict(state)
        self.current_episode = {
            "state": state,
            "action": action,
        }
        return [index]

    def _observe(self, next_state, reward, done):
        if not self.current_episode:
            raise ValueError("Current episode is expected to be non-empty")

        self.current_episode.update({
            "next_state": next_state,
            "reward": reward,
            "done": done
        })

        self.agent.episode = self.current_episode
        if not self.eval_mode:
            self.agent.replay_buffer.add(**self.current_episode)
            if self.agent.t >= self.agent.training_starts and len(self.agent.replay_buffer) >= self.agent.batch_size:
                self.agent.update()
        self.current_episode = {}

    def _extract_state(self, observation):
        return self.agent.embeddings[observation['user'][0]]
