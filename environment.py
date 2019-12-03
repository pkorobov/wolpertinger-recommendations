import numpy as np
from gym import spaces
from recsim import document
from recsim import user
from recsim.choice_model import AbstractChoiceModel

SEED = 1
np.random.seed(SEED)

W = np.array([[.100, .800],
              [.050, .900]])

# W = np.array([[.010, .700, .800],
#               [.020, .850, .900],
#               [.050, .800, .950]])

DOC_NUM = W.shape[0]

P_EXIT_ACCEPTED = 0.1
P_EXIT_NOT_ACCEPTED = 0.2




class Document(document.AbstractDocument):

    def __init__(self, doc_id):
        super(Document, self).__init__(doc_id)
        
    def create_observation(self):
        return np.array([self._doc_id])

    @staticmethod
    def observation_space():
        return spaces.Discrete(DOC_NUM)
  
    def __str__(self):
        return "Document #{}".format(self._doc_id)


class DocumentSampler(document.AbstractDocumentSampler):

    def __init__(self, doc_ctor=Document):
        super(DocumentSampler, self).__init__(doc_ctor, seed=SEED)
        self._doc_count = 0
        
    def sample_document(self):
        doc = self._doc_ctor(self._doc_count % DOC_NUM)
        self._doc_count += 1
        return doc


class UserState(user.AbstractUserState):
    
    def __init__(self, user_id, current, active_session=True):
        self.user_id = user_id
        self.current = current
        self.active_session = active_session

    def create_observation(self):
        return np.array([self.current])

    def __str__(self):
        return "User #{}".format(self.user_id)

    @staticmethod
    def observation_space():
        return spaces.Discrete(DOC_NUM)

    def score_document(self, doc_obs):
        return W[self.current, doc_obs[0]]


class StaticUserSampler(user.AbstractUserSampler):

    def __init__(self, user_ctor=UserState):
        super(StaticUserSampler, self).__init__(user_ctor, seed=SEED)
        self.user_count = 0

    def sample_user(self):
        self.user_count += 1
        sampled_user = self._user_ctor(self.user_count, np.random.randint(DOC_NUM))
        return sampled_user


class Response(user.AbstractResponse):

    def __init__(self, accept=False):
        self.accept = accept

    def create_observation(self):
        return np.array([int(self.accept)])

    @classmethod
    def response_space(cls):
        return spaces.Discrete(2)


class UserChoiceModel(AbstractChoiceModel):
    def __init__(self):
        super(UserChoiceModel, self).__init__()
        self._score_no_click = P_EXIT_ACCEPTED

    def score_documents(self, user_state, doc_obs):
        if len(doc_obs) != 1:
            raise ValueError("Expecting single document, but got: {}".format(doc_obs))
        self._scores = np.array([user_state.score_document(doc) for doc in doc_obs])

    def choose_item(self):
        if np.random.random() < self.scores[0]:
            return 0


class UserModel(user.AbstractUserModel):

    def __init__(self):
        super(UserModel, self).__init__(Response, StaticUserSampler(), 1)
        self.choice_model = UserChoiceModel()

    def simulate_response(self, slate_documents):
        if len(slate_documents) != 1:
            raise ValueError("Expecting single document, but got: {}".format(slate_documents))

        responses = [self._response_model_ctor() for _ in slate_documents]

        self.choice_model.score_documents(self._user_state, [doc.create_observation() for doc in slate_documents])
        selected_index = self.choice_model.choose_item()

        if selected_index is not None:
            responses[selected_index].accept = True

        return responses

    def update_state(self, slate_documents, responses):
        if len(slate_documents) != 1:
            raise ValueError("Expecting single document, but got: {}".format(slate_documents))

        response = responses[0]
        doc = slate_documents[0]
        if response.accept:
            self._user_state.current = doc.doc_id()
            self._user_state.active_session = bool(np.random.binomial(1, 1 - P_EXIT_ACCEPTED))
        else:
            self._user_state.current = np.random.choice(DOC_NUM)
            self._user_state.active_session = bool(np.random.binomial(1, 1 - P_EXIT_NOT_ACCEPTED))

    def is_terminal(self):
        """Returns a boolean indicating if the session is over."""
        return not self._user_state.active_session


def clicked_reward(responses):
    reward = 0.0
    for response in responses:
        if response.accept:
            reward += 1
    return reward
