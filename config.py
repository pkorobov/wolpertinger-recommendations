import numpy as np
import json
import pickle
import platform
from pathlib import Path
from itertools import product

ENV_PARAMETERS = None
AGENT_PARAMETERS = None
DOC_NUM = None
MAX_TOTAL_STEPS = None
W = None
MOST_POPULAR = None
REINIT_STEPS = None
EMBEDDINGS = None
PATH = Path("/Users" if platform.system() == 'Darwin' else "/home") / "p.korobov/data/netflix/matrix_env"
OPTIMAL_ACTIONS = None
OPTIMAL_PROBAS = None


def init_config(param_path='parameters.json'):

    global ENV_PARAMETERS, AGENT_PARAMETERS, MAX_TOTAL_STEPS, \
           DOC_NUM, W, AGENT_PARAM_STRINGS, EMBEDDINGS, \
           MAX_ACTION, MIN_ACTION

    with open(param_path) as json_file:
        all_parameters = json.load(json_file)

    ENV_PARAMETERS = all_parameters['env']
    AGENT_PARAMETERS = all_parameters['agent']
    DOC_NUM = ENV_PARAMETERS['doc_num']
    W = np.zeros((DOC_NUM, DOC_NUM))

    if ENV_PARAMETERS['type'] == 'movies':
        with open(PATH / "embeddings_dict.pkl", "rb") as pickle_in:
            emb_dict = pickle.load(pickle_in)
            EMBEDDINGS = np.array([*map(lambda ind: emb_dict[ind], np.arange(len(emb_dict)))])
            EMBEDDINGS = EMBEDDINGS[:DOC_NUM]

    if ENV_PARAMETERS['type'] == 'movies_uniform_embeddings':
        np.random.seed(0)
        DOC_NUM = 100
        d = 4
        axis = np.linspace(0, 1, np.power(DOC_NUM, 1 / d).astype(int) + 1)
        EMBEDDINGS = np.array(list(product(*([axis] * d))))
        idx = sorted(np.random.choice(EMBEDDINGS.shape[0], size=DOC_NUM))
        EMBEDDINGS = EMBEDDINGS[idx, :]

    if ENV_PARAMETERS['type'] == 'movies_random_embeddings':
        np.random.seed(0)
        EMBEDDINGS = np.random.uniform(DOC_NUM, 4)


def init_w():
    global W, EMBEDDINGS, OPTIMAL_ACTIONS, OPTIMAL_PROBAS

    with open(PATH / "W_matrix.pkl", "rb") as pickle_in:
        W = pickle.load(pickle_in)
        W = W[:DOC_NUM, :DOC_NUM]
        OPTIMAL_ACTIONS = W.argmax(axis=0)
        OPTIMAL_PROBAS = W.max(axis=0)

    # with open(PATH / "embeddings_dict.pkl", "rb") as pickle_in:
        #     emb_dict = pickle.load(pickle_in)
        #     EMBEDDINGS = np.array([*map(lambda ind: emb_dict[ind], np.arange(len(emb_dict)))])
        #     EMBEDDINGS = EMBEDDINGS[:DOC_NUM]

