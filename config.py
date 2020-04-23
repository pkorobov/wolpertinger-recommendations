import numpy as np
import json
import pickle
import platform
from pathlib import Path

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


def init_config(param_path='parameters.json'):

    global ENV_PARAMETERS, AGENT_PARAMETERS, MAX_TOTAL_STEPS, \
           DOC_NUM, W, AGENT_PARAM_STRINGS, EMBEDDINGS

    with open(param_path) as json_file:
        all_parameters = json.load(json_file)

    ENV_PARAMETERS = all_parameters['env']
    AGENT_PARAMETERS = all_parameters['agent']

    def generate_param_strings(agent_params):
        return [f"{params['backend']}, " \
                f"clr={params['critic_lr']}, " \
                f"alr={params['actor_lr']}," \
                f"awd={params['actor_weight_decay']}, " \
                f"cwd={params['critic_weight_decay']}, " \
                f"tau={params['tau']}, " \
                f"eps={params['eps']}" if params['agent'] == 'Wolpertinger' else ''
                for params in agent_params]

    AGENT_PARAM_STRINGS = generate_param_strings(AGENT_PARAMETERS)

    DOC_NUM = ENV_PARAMETERS['doc_num']
    W = np.zeros((DOC_NUM, DOC_NUM))

    with open(PATH / "embeddings_dict.pkl", "rb") as pickle_in:
        emb_dict = pickle.load(pickle_in)
        EMBEDDINGS = np.array([*map(lambda ind: emb_dict[ind], np.arange(len(emb_dict)))])
        EMBEDDINGS = EMBEDDINGS[:DOC_NUM]


def init_w():
    global W, EMBEDDINGS, OPTIMAL_ACTIONS
    if ENV_PARAMETERS['type'] == 'movies':
        with open(PATH / "W_matrix_contrasted_100.pkl", "rb") as pickle_in:
            W = pickle.load(pickle_in)
            W = W[:DOC_NUM, :DOC_NUM]
            OPTIMAL_ACTIONS = W.argmax(axis=0)
        with open(PATH / "embeddings_dict.pkl", "rb") as pickle_in:
            emb_dict = pickle.load(pickle_in)
            EMBEDDINGS = np.array([*map(lambda ind: emb_dict[ind], np.arange(len(emb_dict)))])
            EMBEDDINGS = EMBEDDINGS[:DOC_NUM]
