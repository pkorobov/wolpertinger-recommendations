import os

import numpy as np
import pandas as pd
import torch
from gym import spaces
from recsim import user
from recsim.choice_model import AbstractChoiceModel

from environment.netflix.model.attentive import AttentiveRecommender
from environment.netflix.model.datasets import feature_movies, feature_months, feature_ratings
from environment.netflix.model.prepare_model import find_data_parts, read_data_part, read_movie_indexes, START_DATE, END_DATE
from environment.netflix.preprocess import Rating
from environment.netflix.simulator.config import config, SEED


def read_data():
    partitions = config["environment"]["partitions"]
    data_parts = list(find_data_parts(config))[-partitions:]
    return pd.concat([read_data_part(part) for part in data_parts])


class UserState(user.AbstractUserState):

    def __init__(self, user_id):
        self.user_id = user_id
        self.ratings = []

    def create_observation(self):
        return np.array([self.user_id])

    def update(self, movie, rating, date):
        self.ratings.append(Rating(movie.doc_id(), rating, date))

    def __str__(self):
        return "User#{}".format(self.user_id)

    @staticmethod
    def observation_space():
        return spaces.Discrete(config["environment"]["num_users"])


class SessionProvider(object):

    def __init__(self, clock, max_interval_days=7):
        self.data = read_data()
        self.max_interval_days = max_interval_days
        self.clock = clock

        self.sessions_iter = None

        self.current_user = None
        self.user_states = {}

        self.current_session_dates = None
        self.current_date_index = None

    def reset(self):
        print("Resetting session provider")
        sessions = []
        for user_id, user_sessions in self.data["ratings"].map(self.sessionize).items():
            for session in user_sessions:
                sessions.append((user_id, session))
        print("Sessions found: {}".format(len(sessions)))

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
        if not self.has_next_date():
            raise ValueError("Session ended, there is no next date")
        self.current_date_index += 1
        self.clock.set_date(self.get_current_date())

    def to_next_session(self):
        try:
            current_user_id, self.current_session_dates = next(self.sessions_iter)
        except StopIteration as si:
            raise ValueError("Out of sessions! Oomph")

        self.current_user = self.user_states.get(current_user_id)
        if self.current_user is None:
            self.current_user = UserState(current_user_id)
            self.user_states[current_user_id] = self.current_user

        print("New session", self.current_user, len(self.current_session_dates))
        self.current_date_index = 0
        self.clock.set_date(self.get_current_date())

    def get_current_user(self):
        return self.current_user

    def get_current_date(self):
        return self.current_session_dates[self.current_date_index]

    def has_next_date(self):
        return self.current_date_index < len(self.current_session_dates) - 1


class RatingResponse(user.AbstractResponse):

    def __init__(self, rating=None):
        self.rating = rating

    def create_observation(self):
        return np.array([self.rating])

    @classmethod
    def response_space(cls):
        return spaces.Box(-1, 1)


class UserSampler(user.AbstractUserSampler):

    def __init__(self, session_provider):
        self.session_provider = session_provider
        super(UserSampler, self).__init__(None, seed=SEED)

    def sample_user(self):
        self.session_provider.to_next_session()
        return self.session_provider.get_current_user()

    def reset_sampler(self):
        self.session_provider.reset()


class UserChoiceModel(AbstractChoiceModel):

    def __init__(self, clock, temperature=1.0):
        super(UserChoiceModel, self).__init__()
        self.clock = clock
        self.dataset_args = dict(movie_indexes=read_movie_indexes(config), start_date=START_DATE, end_date=END_DATE)
        self.temperature = temperature

        self.precomputed_features = {
            "movies": feature_movies,
            "months": feature_months,
            "ratings": feature_ratings
        }

        self.model = AttentiveRecommender(config)
        self.model.load_state_dict(torch.load(os.path.join(os.environ["config_path"], "model.params")))

    def score_documents(self, user_state, docs):
        ratings = [
            user_state.ratings + [Rating(doc.doc_id(), 0.0, self.clock.get_current_date())]
            for doc in docs
        ]

        precomputed = self.precompute_features(ratings)
        featurized = self.featurize(precomputed, len(docs))

        self._scores = self.model(featurized).detach().numpy().flatten()

    def precompute_features(self, ratings):
        precomputed = {}
        for feature in config["features"]:
            if not feature.get("precompute", False):
                continue

            feature_name = feature["name"]
            feature_fun = self.precomputed_features[feature_name]
            precomputed[feature_name] = [feature_fun(r, feature, **self.dataset_args) for r in ratings]
        return precomputed

    def featurize(self, data, slate_size):
        features = []
        for j, feature in enumerate(config["features"]):
            if "source" in feature:
                feature_data = self._target_data(data[feature["source"]])
            else:
                feature_data = self._sequence_data(data[feature["name"]])

            padding = feature.get("padding", 0)
            if padding:
                feature_data = [self._pad_sequence(f, padding, 1) for f in feature_data]

            features.append(np.array(feature_data).reshape((slate_size, -1)))

        return torch.from_numpy(np.concatenate(features, axis=1)).float()

    def _sequence_data(self, precomputed):
        return [p[:-1] for p in precomputed]

    def _target_data(self, precomputed):
        return [p[-1] for p in precomputed]

    def _pad_sequence(self, sequence, padding, dim):
        pad_with = [0] * dim if dim > 1 else 0
        return [pad_with] * (padding - len(sequence)) + sequence[-padding:]

    def choose_item(self):
        exp_p = np.exp(self.scores / self.temperature)
        p = exp_p / exp_p.sum()
        return np.random.choice(np.arange(len(self.scores)), p=p)


class UserModel(user.AbstractUserModel):

    def __init__(self, user_sampler, slate_size, choice_model, session_provider):
        super(UserModel, self).__init__(RatingResponse, user_sampler, slate_size)
        self.choice_model = choice_model
        self.session_provider = session_provider

    def simulate_response(self, slate_documents):
        responses = [self._response_model_ctor() for _ in slate_documents]

        self.choice_model.score_documents(self._user_state, slate_documents)
        selected_index = self.choice_model.choose_item()

        if selected_index is not None:
            selected_item_score = self.choice_model.scores[selected_index]
            responses[selected_index].rating = np.round((1.0 + selected_item_score) * 2.0) + 1

        return responses

    def update_state(self, slate_documents, responses):
        for j, response in enumerate(responses):
            if response.rating is None:
                continue
            movie_id = slate_documents[j]
            self._user_state.update(movie_id, response.rating, self.session_provider.get_current_date())

        if self.session_provider.has_next_date():
            self.session_provider.to_next_date()

    def is_terminal(self):
        return not self.session_provider.has_next_date()
