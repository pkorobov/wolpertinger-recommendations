import numpy as np
import pandas as pd
from gym import spaces
from recsim import user
from recsim.choice_model import AbstractChoiceModel

from environment.netflix.model.prepare_model import find_data_parts, read_data_part
from environment.netflix.simulator.config import config, SEED


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


class Response(user.AbstractResponse):

    def __init__(self, rating=0):
        self.rating = rating

    def create_observation(self):
        return np.array([self.rating])

    @classmethod
    def response_space(cls):
        return spaces.Box(-1, 1)


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
        self.session_provider = session_provider
        super(UserSampler, self).__init__(user_ctor, seed=SEED)

    def sample_user(self):
        self.session_provider.to_next_session()
        user_id = self.session_provider.get_current_user()
        date = self.session_provider.get_current_date()
        sampled_user = self._user_ctor(user_id, date)
        return sampled_user

    def reset_sampler(self):
        self.session_provider.reset()


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
        if self.session_provider.get_next_date() is not None:
            self.session_provider.to_next_date()
            self._user_state.date = self.session_provider.get_current_date()

    def is_terminal(self):
        return self.session_provider.get_next_date() is None


