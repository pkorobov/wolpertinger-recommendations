import numpy as np
import json
import pickle

ENV_PARAMETERS = None
AGENT_PARAMETERS = None
DOC_NUM = None
MAX_TOTAL_STEPS = None
W = None
MOST_POPULAR = None
REINIT_STEPS = None
AGENT_PARAM_STRINGS = None
EMBEDDINGS = None

def init_config(param_path='parameters.json'):

    global ENV_PARAMETERS, AGENT_PARAMETERS, MAX_TOTAL_STEPS, \
           DOC_NUM, W, AGENT_PARAM_STRINGS, EMBEDDINGS

    with open(param_path) as json_file:
        all_parameters = json.load(json_file)

    ENV_PARAMETERS = all_parameters['env']
    AGENT_PARAMETERS = all_parameters['agent']
    AGENT_PARAM_STRINGS = all_parameters['agent_param_strings']

    DOC_NUM = ENV_PARAMETERS['doc_num']
    MAX_TOTAL_STEPS = ENV_PARAMETERS['max_total_steps']
    W = np.zeros((DOC_NUM, DOC_NUM))

    with open("/home/p.korobov/data/netflix/matrix_env/embeddings_dict.pkl", "rb") as pickle_in:
        emb_dict = pickle.load(pickle_in)
        EMBEDDINGS = np.array([*map(lambda ind: emb_dict[ind], np.arange(len(emb_dict)))])
        EMBEDDINGS = EMBEDDINGS[:DOC_NUM]

def init_w(reinit=False):
    global W, EMBEDDINGS
    if ENV_PARAMETERS['type'] == 'movies':
        with open("/home/p.korobov/data/netflix/matrix_env/W_matrix.pkl", "rb") as pickle_in:
            W = pickle.load(pickle_in)
            W = W[:DOC_NUM, :DOC_NUM]
        with open("/home/p.korobov/data/netflix/matrix_env/embeddings_dict.pkl", "rb") as pickle_in:
            emb_dict = pickle.load(pickle_in)
            EMBEDDINGS = np.array([*map(lambda ind: emb_dict[ind], np.arange(len(emb_dict)))])
            EMBEDDINGS = EMBEDDINGS[:DOC_NUM]
#
# def init_w(reinit=False):
#
#     global MOST_POPULAR, REINIT_STEPS
#     if ENV_PARAMETERS['type'] == 'alternating':
#         W[:, :] = np.random.uniform(0.0, 0.2, (DOC_NUM, DOC_NUM))
#         if not reinit:
#             REINIT_STEPS = ENV_PARAMETERS['reinit_steps']
#             MOST_POPULAR = ENV_PARAMETERS['most_popular']
#             W[:, MOST_POPULAR] = np.random.uniform(0.9, 1.0, DOC_NUM)
#         else:
#             new_most_popular = np.random.randint(DOC_NUM)
#             while new_most_popular == MOST_POPULAR:
#                 new_most_popular = np.random.randint(DOC_NUM)
#             MOST_POPULAR = new_most_popular
#             W[:, MOST_POPULAR] = np.random.uniform(0.9, 1.0, DOC_NUM)
#
#     if reinit:
#         return
#
#     if ENV_PARAMETERS['type'] == 'one_most_popular':
#         W[:, :] = np.random.uniform(0.0, 0.2, (DOC_NUM, DOC_NUM))
#         MOST_POPULAR = ENV_PARAMETERS['most_popular']
#         W[:, MOST_POPULAR] = np.random.uniform(0.9, 1.0, DOC_NUM)
#
#     if ENV_PARAMETERS['type'] == 'permutation':
#         W[:, :] = (np.ones((DOC_NUM, DOC_NUM)) - np.eye(DOC_NUM)) * \
#                    np.random.uniform(0.0, 0.2, (DOC_NUM, DOC_NUM)) + \
#                    np.diag(np.random.uniform(0.9, 1.0, DOC_NUM))
#         W[:, :] = W[:, np.random.permutation(DOC_NUM)]
