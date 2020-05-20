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
AGENT_PARAM_STRINGS = None
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

    def generate_param_strings(agent_params):
        params_to_display = ['backbone', 'lr', 'critic_lr', 'actor_lr',
                             'actor_weight_decay', 'critic_weight_decay', 'eps']

        param_string = [""] * len(agent_params)
        for i, params_dict in enumerate(agent_params):
            for parameter_name in params_to_display:
                if parameter_name in params_dict:
                    param_string[i] += f"{parameter_name}={params_dict[parameter_name]}, "
            param_string[i] = param_string[i][:-2]
        return param_string

    AGENT_PARAM_STRINGS = generate_param_strings(AGENT_PARAMETERS)

    DOC_NUM = ENV_PARAMETERS['doc_num']
    W = np.zeros((DOC_NUM, DOC_NUM))

    if ENV_PARAMETERS['type'] == 'movies':
        with open(PATH / "embeddings_dict.pkl", "rb") as pickle_in:
            emb_dict = pickle.load(pickle_in)
            EMBEDDINGS = np.array([*map(lambda ind: emb_dict[ind], np.arange(len(emb_dict)))])
            EMBEDDINGS = EMBEDDINGS[:DOC_NUM]

    if ENV_PARAMETERS['type'] == 'movies_uniform_embeddings':
        DOC_NUM = 100
        d = 4
        axis = np.linspace(0, 1, np.power(DOC_NUM, 1 / d).astype(int) + 1)
        EMBEDDINGS = np.array(list(product(*([axis] * d))))
        idx = sorted(np.random.choice(EMBEDDINGS.shape[0], size=DOC_NUM))
        EMBEDDINGS = EMBEDDINGS[idx, :]

    if ENV_PARAMETERS['type'] == 'movies_random_embeddings':
        EMBEDDINGS = np.random.uniform(DOC_NUM, 4)

def init_w():
    global W, EMBEDDINGS, OPTIMAL_ACTIONS, OPTIMAL_PROBAS

    if ENV_PARAMETERS['type'] == 'movies':
        with open(PATH / "W_matrix.pkl", "rb") as pickle_in:
            W = pickle.load(pickle_in)
            W = W[:DOC_NUM, :DOC_NUM]
            OPTIMAL_ACTIONS = W.argmax(axis=0)
            OPTIMAL_PROBAS = W.max(axis=0)

    # with open(PATH / "embeddings_dict.pkl", "rb") as pickle_in:
        #     emb_dict = pickle.load(pickle_in)
        #     EMBEDDINGS = np.array([*map(lambda ind: emb_dict[ind], np.arange(len(emb_dict)))])
        #     EMBEDDINGS = EMBEDDINGS[:DOC_NUM]

