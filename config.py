import numpy as np
import json

ENV_PARAMETERS = None
AGENT_PARAMETERS = None
DOC_NUM = None
MAX_TOTAL_STEPS = None
W = None
MOST_POPULAR = None
REINIT_STEPS = None
AGENT_PARAM_STRINGS = None


def init_config(param_path='parameters/permutation_1.json'):

    global ENV_PARAMETERS, AGENT_PARAMETERS, MAX_TOTAL_STEPS, \
           DOC_NUM, W, AGENT_PARAM_STRINGS

    with open(param_path) as json_file:
        all_parameters = json.load(json_file)

    ENV_PARAMETERS = all_parameters['env']
    AGENT_PARAMETERS = all_parameters['agent']
    AGENT_PARAM_STRINGS = all_parameters['agent_param_strings']

    DOC_NUM = ENV_PARAMETERS['doc_num']
    MAX_TOTAL_STEPS = ENV_PARAMETERS['max_total_steps']
    W = np.zeros((DOC_NUM, DOC_NUM))


def init_w(reinit=False):

    global MOST_POPULAR, REINIT_STEPS
    if ENV_PARAMETERS['type'] == 'alternating':
        W[:, :] = np.random.uniform(0.0, 0.2, (DOC_NUM, DOC_NUM))
        if not reinit:
            REINIT_STEPS = ENV_PARAMETERS['reinit_steps']
            MOST_POPULAR = ENV_PARAMETERS['most_popular']
            W[:, MOST_POPULAR] = np.random.uniform(0.9, 1.0, DOC_NUM)
        else:
            new_most_popular = np.random.randint(DOC_NUM)
            while new_most_popular == MOST_POPULAR:
                new_most_popular = np.random.randint(DOC_NUM)
            MOST_POPULAR = new_most_popular
            W[:, MOST_POPULAR] = np.random.uniform(0.9, 1.0, DOC_NUM)

    if reinit:
        return

    if ENV_PARAMETERS['type'] == 'one_most_popular':
        W[:, :] = np.random.uniform(0.0, 0.2, (DOC_NUM, DOC_NUM))
        MOST_POPULAR = ENV_PARAMETERS['most_popular']
        W[:, MOST_POPULAR] = np.random.uniform(0.9, 1.0, DOC_NUM)

    if ENV_PARAMETERS['type'] == 'permutation':
        W[:, :] = (np.ones((DOC_NUM, DOC_NUM)) - np.eye(DOC_NUM)) * \
                   np.random.uniform(0.0, 0.2, (DOC_NUM, DOC_NUM)) + \
                   np.diag(np.random.uniform(0.9, 1.0, DOC_NUM))
        W[:, :] = W[:, np.random.permutation(DOC_NUM)]


# permutation
parameters = {
              "noise_sigma": 0.05,  # was 0.1
              "critic_lr": 1e-3,
              "actor_lr": 1e-3,  # 1e-4
              "soft_tau": 1e-3,
              "hidden_dim": 256,
              "batch_size": 128,
              "buffer_size": 20000,
              "gamma": 0.8,
              "actor_weight_decay": 0.0001,  # 1e-3
              "critic_weight_decay": 0.001,
              "eps": 1e-2  # 1e-1
            }
