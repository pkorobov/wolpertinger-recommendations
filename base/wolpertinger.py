from base.ddpg import DDPG
import matrix_env
import config
import gym
from gym.core import Env
import plots
import numpy as np


class Wolpertinger(DDPG):
    def __init__(self, state_dim, action_dim, env,
                 batch_size=128, gamma=0.99, min_value=-np.inf, max_value=np.inf,
                 k_ratio=0.1, training_starts=100, eps=1e-2, embeddings=None, **kwargs):

        super(Wolpertinger, self).__init__(state_dim, action_dim,
                                                batch_size=batch_size, gamma=gamma,
                                                min_value=min_value, max_value=max_value,
                                                **kwargs)
        self.k = max(1, int(action_dim * k_ratio))
        self.training_starts = training_starts
        self.eps = eps
        self.episode = None
        self.last_proto = None

    def predict(self, state):

        proto_action = super().predict(state)
        proto_action = proto_action.clip(0, 1)

        actions = np.eye(self.action_dim)[np.lexsort((np.random.random(self.action_dim), proto_action))[-self.k:]]
        states = np.tile(state, [len(actions), 1])  # make all the state-action pairs for the critic
        q_values = self.critic.get_q_values(states, actions)
        max_index = np.argmax(q_values)  # find the index of the pair with the maximum value
        action, q_value = actions[max_index], q_values[max_index]

        # logging
        self.last_proto = proto_action
        if self.summary_writer:
            if self.t % 500 == 0:
                q_values = np.hstack([self.compute_q_values(i) for i in range(config.DOC_NUM)]).T
                q_values_target = np.hstack([self.compute_q_values_target(i) for i in range(config.DOC_NUM)]).T
                actions = np.vstack([super(Wolpertinger, self)
                                    .predict(np.eye(config.DOC_NUM)[i], with_noise=False)
                                     for i in range(config.DOC_NUM)])

                plots.heatmap(self.summary_writer, q_values.round(3), 'Q_values/main', self.t)
                plots.heatmap(self.summary_writer, q_values_target.round(3), 'Q_values/target', self.t)
                plots.heatmap(self.summary_writer, actions.round(3), 'policy/actions', self.t)
                if config.DOC_NUM == 2:
                    plots.plot_2d_function(self.summary_writer,
                                           lambda x, y: self.critic.get_q_values([[0.0, 1.0]], [[x, y]]),
                                           'Q_values/q_values_s=0', self.t)
        return action

    def compute_actions(self, state_num=0, a=None, dim=None):
        if dim is None:
            dim = self.action_dim
        if a is None:
            a = np.eye(dim, self.action_dim)
        s = np.zeros((dim, self.action_dim))
        s[:, state_num] = 1
        q_vector = self.critic.get_q_values(s, a)
        return q_vector

    def compute_q_values(self, state_num=0, a=None, dim=None):
        if dim is None:
            dim = self.action_dim
        if a is None:
            a = np.eye(dim, self.action_dim)
        s = np.zeros((dim, self.action_dim))
        s[:, state_num] = 1
        q_vector = self.critic.get_q_values(s, a)
        return q_vector

    def compute_q_values_target(self, state_num=0, a=None, dim=None):
        if dim is None:
            dim = self.action_dim
        if a is None:
            a = np.eye(dim, self.action_dim)
        s = np.zeros((dim, self.action_dim))
        s[:, state_num] = 1
        q_vector = self.critic_target.get_q_values(s, a)
        return q_vector
