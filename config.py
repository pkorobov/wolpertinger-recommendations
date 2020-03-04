import numpy as np
from base.ddpg import GaussNoise
import json

DOC_NUM = 10
W = np.zeros((DOC_NUM, DOC_NUM))

with open('env_parameters.json') as json_file:
    ENV_PARAMETERS = json.load(json_file)

def init_w():
    if ENV_PARAMETERS['kind'] == 'one_most_popular':
        W[:, :] = np.random.uniform(0.0, 0.2, (DOC_NUM, DOC_NUM))
        MOST_POPULAR = ENV_PARAMETERS['most_popular']
        W[:, MOST_POPULAR] = np.random.uniform(0.9, 1.0, DOC_NUM)
    if ENV_PARAMETERS['kind'] == 'permutation':
        W[:, :] = (np.ones((DOC_NUM, DOC_NUM)) - np.eye(DOC_NUM)) * np.random.uniform(0.0, 0.2, (DOC_NUM, DOC_NUM)) + \
            np.diag(np.random.uniform(0.9, 1.0, DOC_NUM))
        W[:, :] = W[:, np.random.permutation(DOC_NUM)]

# parameters = {'action_dim': DOC_NUM,
#               'state_dim': DOC_NUM,
#               'noise': GaussNoise(sigma=0.1),
#               'critic_lr': 1e-3,
#               'actor_lr': 1e-3,  # higher learning rate (was 1e-4)
#               'soft_tau': 1e-3,
#               'hidden_dim': 64,  # bigger networks (was 16)
#               'batch_size': 128,
#               'buffer_size': 20000,  # bigger buffer (was 1000)
#               'actor_weight_decay': 0.,  # regularization
#               'critic_weight_decay': 0.001,
#               'gamma': 0.8}

# shift env
parameters = {'action_dim': DOC_NUM,
              'state_dim': DOC_NUM,
              'noise': GaussNoise(sigma=0.05),
              'critic_lr': 1e-3,
              'actor_lr': 1e-3,
              'soft_tau': 1e-3,
              'hidden_dim': 256,
              'batch_size': 128,
              'buffer_size': 20000,
              'gamma': 0.8,
              # 'actor_weight_decay': 0.05,
              'actor_weight_decay': 0.0001,
              'critic_weight_decay': 0.001,
              'eps': 1e-2
              # 'training_starts': 1000
              }


# static env
# parameters = {'action_dim': DOC_NUM,
#               'state_dim': DOC_NUM,
#               'noise': GaussNoise(sigma=0.1),
#               'critic_lr': 1e-3,
#               'actor_lr': 1e-3,  # higher learning rate (was 1e-4)
#               'soft_tau': 1e-3,
#               'hidden_dim': 64,  # bigger networks (was 16)
#               'batch_size': 128,
#               'buffer_size': 20000,  # bigger buffer (was 1000)
#               'actor_weight_decay': 0.001,  # regularization
#               'critic_weight_decay': 0.001,
#               'gamma': 0.8}

# shift env
# parameters = {'action_dim': DOC_NUM,
#               'state_dim': DOC_NUM,
#               'noise': GaussNoise(sigma=0.05),
#               'critic_lr': 1e-3,
#               'actor_lr': 1e-3,
#               'soft_tau': 1e-3,
#               'hidden_dim': 256,
#               'batch_size': 128,
#               'buffer_size': 20000,
#               'gamma': 0.8,
#               # 'actor_weight_decay': 0.05,
#               'actor_weight_decay': 0.001,
#               'critic_weight_decay': 0.001,
#               'eps': 1e-1
#               # 'training_starts': 1000
#               }

# alternating
# parameters = {'action_dim': DOC_NUM,
#               'state_dim': DOC_NUM,
#               'noise': GaussNoise(sigma=0.1),
#               'critic_lr': 1e-3,
#               'actor_lr': 1e-3,
#               'soft_tau': 1e-3,
#               'hidden_dim': 256,
#               'batch_size': 128,
#               'buffer_size': 1000,
#               'gamma': 0.8,
#               'actor_weight_decay': 0.1,
#               'critic_weight_decay': 0.1,
#               'init_w_actor': 3e-3,
#               }
