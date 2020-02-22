import pandas as pd

from collections import defaultdict
from recsim import agent
from environment.netflix.preprocess import Rating


class LoggingAgentWrapper(agent.AbstractEpisodicRecommenderAgent):

    def __init__(self, action_space, wrapped, clock):
        super(LoggingAgentWrapper, self).__init__(action_space)
        self.wrapped = wrapped
        self.clock = clock

        self.user = None
        self.slate = None
        self.date = None

        self.episode_ratings = []
        self.logged_data = defaultdict(list)

    def begin_episode(self, observation=None):
        self.user = observation["user"][0]
        self.date = self.clock.get_current_date()
        self.slate = self.wrapped.begin_episode(observation)
        return self.slate

    def step(self, reward, observation):
        self.log_rating(observation["response"])

        self.user = observation["user"][0]
        self.date = self.clock.get_current_date()
        self.slate = self.wrapped.step(reward, observation)

        return self.slate

    def end_episode(self, reward, observation=None):
        self.log_rating(observation["response"])
        self.logged_data[self.user] += self.episode_ratings
        self.episode_ratings = []

    def log_rating(self, responses):
        for j, movie in enumerate(self.slate):
            if responses[j][0] is not None:
                self.episode_ratings += [Rating(movie, responses[j][0], self.date)]

    def write_logged_data(self, path):
        data = pd.Series(self.logged_data).to_frame("ratings")
        pd.to_pickle(data, path)
