from .ddpg import DDPG
from .sac import SAC
from .td3 import TD3
import matrix_env
import config
import gym
from gym.core import Env
import plots
import numpy as np
import faiss
import torch
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_wolpertinger(backbone=DDPG):
    class Wolpertinger(backbone):
        def __init__(self, state_dim, action_dim, env, batch_size=128, gamma=0.99,
                     k_ratio=0.1, training_starts=100, eps=1e-2, embeddings=None, **kwargs):

            super(Wolpertinger, self).__init__(state_dim, action_dim,
                                               batch_size=batch_size, gamma=gamma,
                                               **kwargs)

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
            self.backbone = backbone

        def predict(self, state):

            proto_action = super().predict(state)
            proto_action = proto_action.clip(-1, 1)

            D, I = self.index.search(proto_action[np.newaxis, :].astype(np.float32), self.k)
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

        def target_action(self, next_state):
            if self.backbone == SAC:
                next_action, next_state_log_pi, _ = super().target_action(next_state)
            else:
                next_action = super().target_action(next_state)
            I = self.index.search(next_action.cpu().detach().numpy(), self.k)[1]
            proto_action_neighbours = self.embeddings[I]
            proto_action_neighbours = torch.from_numpy(proto_action_neighbours).to(device)
            next_state_tiled = next_state.unsqueeze(1).repeat(1, self.k, 1)

            if self.backbone == TD3:
                q_values = self.critic_target.Q1(next_state_tiled, proto_action_neighbours).squeeze()
                next_action = torch.from_numpy(self.embeddings[q_values.argmax(dim=-1).cpu().detach()]).to(device)
                noise = (
                        torch.randn_like(next_action) * self.policy_noise
                ).clamp(-self.noise_clip, self.noise_clip)
                next_action = (next_action + noise).clamp(-self.max_action, self.max_action)

            if self.backbone == DDPG:
                q_values = self.critic_target(next_state_tiled, proto_action_neighbours).squeeze()
                next_action = torch.from_numpy(self.embeddings[q_values.argmax(dim=-1).cpu().detach()]).to(device)

            if self.backbone == SAC:
                q_values = torch.min(*self.critic_target(next_state_tiled, proto_action_neighbours)).squeeze()
                next_action = torch.from_numpy(self.embeddings[q_values.argmax(dim=-1)])
                return next_action, next_state_log_pi, _
            return next_action

        def print_base(self):
            for base in self.__class__.__bases__:
                print(base.__name__)
    return Wolpertinger
