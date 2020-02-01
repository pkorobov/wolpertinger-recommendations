import collections
import json
import os

import numpy as np
import pandas as pd

from gym import spaces
from recsim import document
from recsim import user
from recsim.choice_model import AbstractChoiceModel
from recsim.simulator.environment import SingleUserEnvironment

from environment.netflix.model.prepare_model import find_data_parts, read_data_part
from environment.netflix.preprocess import Rating


SEED = 42

with(open(os.environ["config_path"])) as config_file:
    print("loading config from {}".format(os.environ["config_path"]))
    config = json.load(config_file)


def read_data():
    partitions = config["environment"]["partitions"]
    data_parts = list(find_data_parts(config))[-partitions:]
    return pd.concat([read_data_part(part) for part in data_parts])


class SessionProvider(object):

    def __init__(self, max_interval_days=7):
        self.data = read_data()
        self.max_interval_days = max_interval_days

        self.sessions_iter = None

        self.current_user = None
        self.current_session_dates = None
        self.current_date_index = None

        self.reset()

    def reset(self):
        sessions = []
        for user_id, user_sessions in self.data["ratings"].map(self.sessionize).items():
            for session in user_sessions:
                sessions.append((user_id, session))
        self.sessions_iter = iter(sorted(sessions, key=lambda user_session: user_session[1][0]))
        self.to_next_session()

    def sessionize(self, ratings):
        current_session = []
        for rating in ratings:
            if current_session and (rating.date - current_session[-1]).days > self.max_interval_days:
                yield current_session
                current_session = []
            current_session.append(rating.date)
        if current_session:
            yield current_session

    def to_next_date(self):
        if self.get_next_date() is None:
            raise ValueError("Session ended, there is no next date")
        self.current_date_index += 1

    def to_next_session(self):
        self.current_user, self.current_session_dates = next(self.sessions_iter)
        print("New session", self.current_user, len(self.current_session_dates))
        self.current_date_index = 0

    def get_current_user(self):
        return self.current_user

    def get_current_date(self):
        return self.current_session_dates[self.current_date_index]

    def get_next_date(self):
        if self.current_date_index + 1 >= len(self.current_session_dates):
            return None
        return self.current_session_dates[self.current_date_index + 1]


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
        # Todo: get movies for current date
        return np.arange(config["environment"]["num_movies"]) + 1


class UserState(user.AbstractUserState):

    def __init__(self, user_id, date):
        # Todo: Keep previous ratings
        self.user_id = user_id
        self.date = date

    def create_observation(self):
        # Todo: Also provide date
        return np.array([self.user_id])

    def __str__(self):
        return "User#{}@{}".format(self.user_id, self.date)

    @staticmethod
    def observation_space():
        return spaces.Discrete(config["environment"]["num_users"])


class UserSampler(user.AbstractUserSampler):

    def __init__(self, session_provider, user_ctor=UserState):
        super(UserSampler, self).__init__(user_ctor, seed=SEED)
        self.session_provider = session_provider

    def sample_user(self):
        self.session_provider.to_next_session()
        user_id = self.session_provider.get_current_user()
        date = self.session_provider.get_current_date()
        sampled_user = self._user_ctor(user_id, date)
        return sampled_user


class Response(user.AbstractResponse):

    def __init__(self, rating=0):
        self.rating = rating

    def create_observation(self):
        return np.array([self.rating])

    @classmethod
    def response_space(cls):
        return spaces.Box(-1, 1)


class UserChoiceModel(AbstractChoiceModel):
    def __init__(self, session_provider):
        super(UserChoiceModel, self).__init__()
        self.session_provider = session_provider

    def score_documents(self, user_state, docs):
        # Todo: apply pytorch model here
        self._scores = np.zeros(len(docs))

    def choose_item(self):
        # Todo: sample using scores
        return np.random.choice(np.arange(len(self.scores)))


class UserModel(user.AbstractUserModel):

    def __init__(self, user_sampler, slate_size, choice_model, session_provider):
        super(UserModel, self).__init__(Response, user_sampler, slate_size)
        self.choice_model = choice_model
        self.session_provider = session_provider

    def simulate_response(self, slate_documents):
        responses = [self._response_model_ctor() for _ in slate_documents]

        self.choice_model.score_documents(self._user_state, slate_documents)
        selected_index = self.choice_model.choose_item()

        if selected_index is not None:
            responses[selected_index].rating = self.choice_model.scores[selected_index]

        return responses

    def update_state(self, slate_documents, responses):
        # Todo: Append new rating to user state
        pass

    def is_terminal(self):
        terminal = self.session_provider.get_next_date() is None
        if not terminal:
            self.session_provider.to_next_date()
        return terminal


class StaticCandidateSet(document.CandidateSet):

    def __init__(self):
        super().__init__()
        self._documents = {}
        self.observation = None

    def create_observation(self):
        if self.observation is None:
            self.observation = collections.OrderedDict(super().create_observation())
        return self.observation


class NetflixEnvironment(SingleUserEnvironment):

    def __init__(self,
                 user_model,
                 document_sampler,
                 num_candidates,
                 slate_size,
                 resample_documents=True):
        self._candidate_set = StaticCandidateSet()
        super().__init__(user_model, document_sampler, num_candidates, slate_size, resample_documents)
        self._current_documents = self._candidate_set.create_observation()

    def _do_resample_documents(self):
        for _ in range(self._num_candidates):
            self._candidate_set.add_document(self._document_sampler.sample_document())

    def reset(self):
        """Resets the environment and return the first observation.

        Returns:
          user_obs: An array of floats representing observations of the user's
            current state
          doc_obs: An OrderedDict of document observations keyed by document ids
        """
        self._user_model.reset()
        user_obs = self._user_model.create_observation()
        if self._resample_documents:
            self._do_resample_documents()
            self._current_documents = self._candidate_set.create_observation()
        return user_obs, self._current_documents

    def reset_sampler(self):
        """Resets the relevant samplers of documents and user/users."""
        self._document_sampler.reset_sampler()
        self._user_model.reset_sampler()

    def step(self, slate):
        """Executes the action, returns next state observation and reward.

        Args:
          slate: An integer array of size slate_size, where each element is an index
            into the set of current_documents presented

        Returns:
          user_obs: A gym observation representing the user's next state
          doc_obs: A list of observations of the documents
          responses: A list of AbstractResponse objects for each item in the slate
          done: A boolean indicating whether the episode has terminated
        """

        assert (len(slate) <= self._slate_size
                ), 'Received unexpectedly large slate size: expecting %s, got %s' % (
            self._slate_size, len(slate))

        # Get the documents associated with the slate
        # Todo: proper document indexing
        documents = self._candidate_set.get_documents(slate)
        # Simulate the user's response
        responses = self._user_model.simulate_response(documents)

        # Update the user's state.
        self._user_model.update_state(documents, responses)

        # Update the documents' state.
        self._document_sampler.update_state(documents, responses)

        # Obtain next user state observation.
        user_obs = self._user_model.create_observation()

        # Check if reaches a terminal state and return.
        done = self._user_model.is_terminal()

        # Optionally, recreate the candidate set to simulate candidate
        # generators for the next query.
        if self._resample_documents:
            self._do_resample_documents()
            self._current_documents = self._candidate_set.create_observation()

        return user_obs, self._current_documents, responses, done


def ratings_reward(responses):
    return np.sum([response.rating for response in responses])

