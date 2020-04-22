from base.ddpg import DDPG
import matrix_env
import config
import gym
from gym.core import Env
import plots
import numpy as np
import faiss
import torch
import numpy as np

def createWolpertinger(backend=DDPG):
    class Wolpertinger(backend):
        def __init__(self, state_dim, action_dim, env, batch_size=128, gamma=0.99,
                     min_value=-np.inf, max_value=np.inf, k_ratio=0.1, training_starts=100,
                     eps=1e-2, embeddings=None, **kwargs):

            super(Wolpertinger, self).__init__(state_dim, action_dim,
                                               batch_size=batch_size, gamma=gamma,
                                               **kwargs)
            self.k = max(1, int(action_dim * k_ratio))
            self.training_starts = training_starts
            self.eps = eps
            self.episode = None
            self.last_proto = None

            n, d = embeddings.shape
            self.embeddings = embeddings

            self.index = faiss.IndexFlatL2(d)

            if torch.cuda.is_available():
                res = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(res, 0, self.index)

            self.index.add(embeddings.astype(np.float32))
            self.k = max(1, int(n * k_ratio))

        def predict(self, state):

            proto_action = super().predict(state)
            proto_action = proto_action.clip(-1, 1)

            _, I = self.index.search(proto_action[np.newaxis, :].astype(np.float32), self.k)
            actions = self.embeddings[I[0]]
            states = np.tile(state, [len(actions), 1])  # make all the state-action pairs for the critic
            q_values = self.critic.get_q_values(states, actions)
            max_index = np.argmax(q_values)  # find the index of the pair with the maximum value
            action, index = actions[max_index], I[0][max_index]
            return action, index

        def proto_action(self, state, with_noise=False):
            return super().predict(state, with_noise)

        def compute_q_values(self, state_num=0, target=False):
            s = np.tile(self.embeddings[state_num], (self.embeddings.shape[0], 1))
            a = self.embeddings

            if not target:
                return self.critic.get_q_values(s, a)
            return self.critic_target.get_q_values(s, a)

    return Wolpertinger
