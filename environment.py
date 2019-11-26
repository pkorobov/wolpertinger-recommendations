#!/usr/bin/env python
# coding: utf-8

import numpy as np
from gym import spaces
import matplotlib.pyplot as plt
from scipy import stats
from recsim import document
from recsim import user
from recsim.choice_model import MultinomialLogitChoiceModel
from recsim.simulator import environment
from recsim.simulator import recsim_gym

DOC_NUM = 3
np.random.seed(1)
W = np.random.uniform(0, 1, size=(DOC_NUM, DOC_NUM))

choices_history = dict()
click_history = dict()


def generate_W():
    W = np.random.uniform(0, 1, size=(DOC_NUM, DOC_NUM))
    W = W * (np.ones((DOC_NUM, DOC_NUM)) - np.eye(DOC_NUM, DOC_NUM))
    W = W / W.sum(axis=1).reshape(-1, 1)
    return W

W = generate_W()
W = np.array([[0., .8, .01, .09, .1],
              [.1, .8, 0., 0., .1],
              [0., .7, 0.1, 0.1, 0.1],
              [0., .9, .05, 0., .05],
              [0., 1., 0., 0., 0.]])

W = np.array([[.0, 1., .0],
              [.0, 1., 0.0],
              [0.0, 1., 0.0],])

W = np.array([[.1, .8, .1],
              [.05, .9, 0.05],
              [0.025, .95, 0.025],])

# Модель документа
class LTSDocument(document.AbstractDocument):
#    doc_num = DOC_NUM
    def __init__(self, doc_id):
        # doc_id is an integer representing the unique ID of this document
        super(LTSDocument, self).__init__(doc_id)
        
    def create_observation(self):
        return np.array([self._doc_id])

    @staticmethod
    def observation_space():
        return spaces.Discrete(LTSDocument.doc_num)
  
    def __str__(self):
        return "Document #{}".format(self._doc_id)

class LTSDocumentSampler(document.AbstractDocumentSampler):
    def __init__(self, doc_num=10, doc_ctor=LTSDocument, **kwargs):
        doc_ctor.doc_num = doc_num
        super(LTSDocumentSampler, self).__init__(doc_ctor, **kwargs)
        self._doc_count = 0
        self.doc_num = doc_num
        
    def sample_document(self):
        doc_features = {}
        doc_features['doc_id'] = self._doc_count
        
        if self._doc_count < self.doc_num:
            self._doc_count = (self._doc_count + 1) % self.doc_num
        else:
            self._doc_count = 0
        return self._doc_ctor(**doc_features)

# User state and user sampler
class LTSUserState(user.AbstractUserState):
    
    def __init__(self, current, active_session=True):
        self.current = current
        self.active_session = active_session

    def create_observation(self):
        return np.array([self.current,])

    @staticmethod
    def observation_space():
        return spaces.Discrete(LTSUserState.doc_num)

    # scoring function for use in the choice model -- takes probs from user transition matrix W
    def score_document(self, doc_obs):
#       с одним документом в слейте в общем-то и не нужный метод
#       вызывался в score_documents, чтобы оценить все документы слейта
        return W[self.current, doc_obs[0]]

class LTSStaticUserSampler(user.AbstractUserSampler):
    _state_parameters = None

    def __init__(self, user_ctor=LTSUserState, doc_num=10, current=0, **kwargs):
        user_ctor.doc_num = doc_num
        self.doc_num = doc_num
        self._state_parameters = {'current': current}
        super(LTSStaticUserSampler, self).__init__(user_ctor, **kwargs)

    def sample_user(self):
        current = np.random.randint(self.doc_num)
        self._state_parameters['current'] = current
        return self._user_ctor(**self._state_parameters)

#  Response model
class LTSResponse(user.AbstractResponse):

    def __init__(self, accept=False):
        self.accept = accept

    def create_observation(self):
        return {'accept': int(self.accept)}

    @classmethod
    def response_space(cls):
        return spaces.Dict({
            'accept':
                spaces.Discrete(2),
        })

def user_init(self,
              slate_size,
              doc_num,
              seed=0):

    self.doc_num = doc_num
    LTSUserState.doc_num = doc_num
    
    super(LTSUserModel,
          self).__init__(LTSResponse, LTSStaticUserSampler(LTSUserState, seed=seed, doc_num=DOC_NUM), slate_size)
    self.choice_model = UserChoiceModel()

from recsim.choice_model import NormalizableChoiceModel, softmax

class UserChoiceModel(NormalizableChoiceModel):  # pytype: disable=ignored-metaclass
    def __init__(self):
        super(UserChoiceModel, self).__init__()

    def score_documents(self, user_state, doc_obs):

        assert len(doc_obs) == 1, "Slate must be of size 1"

        score = W[user_state.current, doc_obs[0]]
        assert score <= 1, "Click score must be probability"

        self._score_no_click = 1 - score
        self._scores = [score]

def simulate_response(self, slate_documents):
    # List of empty responses
    responses = [self._response_model_ctor() for _ in slate_documents]

    # Get click from of choice model.
    self.choice_model.score_documents(self._user_state, [doc.create_observation() for doc in slate_documents])
    scores = self.choice_model.scores
    selected_index = self.choice_model.choose_item()

    # Populate clicked item.
    if selected_index in click_history:
        click_history[selected_index] += 1
    else:
        click_history[selected_index] = 1

    if selected_index is None:
        return responses
    self._generate_response(slate_documents[selected_index],
                            responses[selected_index])
    return responses

def generate_response(self, doc, response):
    response.accept = True

def update_state(self, slate_documents, responses):
    for doc, response in zip(slate_documents, responses):
        if response.accept:
            self._user_state.active_session = np.random.binomial(1, 0.9)
            return
        else:
            self._user_state.current = np.random.randint(self.doc_num)
            self._user_state.active_session = np.random.binomial(1, 0.8)

def is_terminal(self):
      """Returns a boolean indicating if the session is over."""
      return self._user_state.active_session


LTSUserModel = type("LTSUserModel", (user.AbstractUserModel,),
                    {"__init__": user_init,
                     "is_terminal": is_terminal,
                     "update_state": update_state,
                     "simulate_response": simulate_response,
                     "_generate_response": generate_response})


ltsenv = environment.Environment(
           LTSUserModel(slate_size, doc_num=DOC_NUM),
           LTSDocumentSampler(doc_num=DOC_NUM),
           num_candidates,
           slate_size,
           resample_documents=True)

def clicked_reward(responses): # не вызывается
  reward = 0.0
  for response in responses:
    if response.accept:
      reward += 1
  return reward

np.random.seed(101)

from recsim.agent import AbstractEpisodicRecommenderAgent
from recsim.agents.full_slate_q_agent import FullSlateQAgent
from recsim.agents.full_slate_q_agent import FullSlateQAgent
from recsim.agents.random_agent import RandomAgent
from recsim.simulator import runner_lib

class StaticAgent(AbstractEpisodicRecommenderAgent):
    def __init__(self, observation_space, action_space):
        super(StaticAgent, self).__init__(action_space)

    def step(self, reward, observation):
        return [0]

def create_agent(sess, environment, eval_mode, summary_writer=None):
    kwargs = {
          'observation_space': environment.observation_space,
          'action_space': environment.action_space,
          'eval_mode': eval_mode,
    }
    return FullSlateQAgent(sess, **kwargs)
    #return StaticAgent(**kwargs)

slate_size = 1
num_candidates = DOC_NUM
ltsenv = environment.Environment(
        LTSUserModel(slate_size, doc_num=DOC_NUM),
        LTSDocumentSampler(doc_num=DOC_NUM),
        num_candidates,
        slate_size,
        resample_documents=False)

env = recsim_gym.RecSimGymEnv(ltsenv, clicked_reward)
tmp_base_dir = 'tmp'
episode_log_file_train = 'episodes_train'

import numpy as np
from recsim.simulator import runner_lib
import subprocess
subprocess.run(["rm", "-rf", "tmp"])
subprocess.run(["mkdir", "tmp"])
subprocess.run(["tensorboard", "--logdir", "tmp", " &"])

runner = runner_lib.TrainRunner(
  base_dir=tmp_base_dir,
  create_agent_fn=create_agent,
  env=env,
  episode_log_file=episode_log_file_train,
  max_training_steps=5,
  num_iterations=10000)
runner.run_experiment()

runner = runner_lib.EvalRunner(
  base_dir=tmp_base_dir,
  create_agent_fn=create_agent,
  env=env,
  max_eval_episodes=50,
  test_mode=True)

runner.run_experiment()
