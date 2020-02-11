import collections

import numpy as np
from gym import spaces
from recsim import document

from environment.netflix.model.prepare_model import read_movie_indexes
from environment.netflix.simulator.config import config, SEED


class StaticCandidateSet(document.CandidateSet):

    def __init__(self):
        super().__init__()
        self._documents = {}
        self.observation = None
        self.doc_ids = None

    def create_observation(self):
        if self.observation is None:
            self.observation = collections.OrderedDict(super().create_observation())
            self.doc_ids = list(self.observation)
        return self.observation


class Movie(document.AbstractDocument):

    def __init__(self, movie_id):
        super(Movie, self).__init__(movie_id)
        self.observation = np.array([self._doc_id])

    def create_observation(self):
        return self.observation

    @staticmethod
    def observation_space():
        return spaces.Discrete(config["environment"]["num_movies"])

    def __str__(self):
        return "Movie#{}".format(self._doc_id)


class MovieSampler(document.AbstractDocumentSampler):

    def __init__(self, movie_ctor=Movie):
        self.movie_indexes = read_movie_indexes(config)
        super(MovieSampler, self).__init__(movie_ctor, seed=SEED)
        self.movie_iter = None

    def sample_document(self):
        if self.movie_iter is None:
            self.reset_sampler()
        try:
            movie = self._doc_ctor(next(self.movie_iter))
        except StopIteration:
            self.reset_sampler()
            movie = self._doc_ctor(next(self.movie_iter))
        return movie

    def reset_sampler(self):
        self.movie_iter = iter(self.get_available_movies())

    def get_available_movies(self):
        return self.movie_indexes.keys()