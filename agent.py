from recsim.agent import AbstractEpisodicRecommenderAgent
import tensorflow as tf
from wolpertinger.wolp_agent import *
from stable_baselines.ddpg import MlpPolicy
from gym import spaces
import os

class StaticAgent(AbstractEpisodicRecommenderAgent):

    def __init__(self, action_space, recommended_doc_id):
        super(StaticAgent, self).__init__(action_space)
        self.recommended_doc_id = recommended_doc_id

    def step(self, reward, observation):
        return [self.recommended_doc_id]


class WolpAgent(AbstractEpisodicRecommenderAgent):

    def __init__(self, env, action_space, k_ratio=0.1, policy_kwargs=None,
                 action_noise=None, eval_mode=False, max_actions=1000, writer=None,
                 full_tensorboard_log=False):
        AbstractEpisodicRecommenderAgent.__init__(self, action_space)

        self._observation_space = env.observation_space
        self.agent = WolpertingerAgent(MlpPolicy, env, tau=1e-5, action_noise=action_noise,
                                       policy_kwargs=policy_kwargs, k_ratio=k_ratio,
                                       full_tensorboard_log=full_tensorboard_log)
        self.t = 0
        self.current_episode = {}
        self.eval_mode = eval_mode
        self.writer = writer

        max_tf_checkpoints_to_keep = 10
        with self.agent.get_sess().as_default(), self.agent.get_graph().as_default():
            self._saver = tf.train.Saver(var_list=tf.all_variables(), max_to_keep=max_tf_checkpoints_to_keep)

    def begin_episode(self, observation=None):
        self.agent._reset()
        state = self._extract_state(observation)
        return self._act(state)

    def step(self, reward, observation):
        state = self._extract_state(observation)
        self._observe(state, reward, 0)
        self.t += 1
        return self._act(state)

    def end_episode(self, reward, observation=None):
        state = self._extract_state(observation)
        self._observe(state, reward, 1)

    def _act(self, state):
        action, q_value = self.agent._policy(state)
        self.current_episode = {
            "obs": state,
            "action": action,
        }
        return np.where(action)[0]

    def _observe(self, next_state, reward, done):
        if not self.current_episode:
            raise ValueError("Current episode is expected to be non-empty")

        self.current_episode.update({
            "next_obs": next_state,
            "reward": reward,
            "done": done
        })
        if not self.eval_mode:
            self.agent._store_transition(**self.current_episode)
            if self.agent.replay_buffer.can_sample(self.agent.batch_size):
                self.agent._train_step(self.t, self.writer)
                self.agent._update_target_net()
        self.current_episode = {}

    def _extract_state(self, observation):
        user_space = self._observation_space.spaces['user']
        return spaces.flatten(user_space, observation['user'])

    def bundle_and_checkpoint(self, checkpoint_dir, iteration_number):

        if not tf.gfile.Exists(checkpoint_dir):
            return None
        # Call the Tensorflow saver to checkpoint the graph.

        self._saver.save(
            self.agent.get_sess(),
            os.path.join(checkpoint_dir, 'tf_ckpt'),
            global_step=iteration_number)

        # Checkpoint the out-of-graph replay buffer.
        # self._replay.save(checkpoint_dir, iteration_number)
        bundle_dictionary = {}
        bundle_dictionary['episode_num'] = self._episode_num

        # bundle_dictionary['state'] = self.state
        # bundle_dictionary['training_steps'] = self.training_steps
        return bundle_dictionary

    def unbundle(self, checkpoint_dir, iteration_number, bundle_dictionary):
        # try:
        #     self._replay.load(checkpoint_dir, iteration_number)
        # except tf.errors.NotFoundError:
        #     if not self.allow_partial_reload:
        #         # If we don't allow partial reloads, we will return False.
        #         return False
        #       tf.logging.warning('Unable to reload replay buffer!')
        # if bundle_dictionary is not None:
        #     for key in self.__dict__:
        #         if key in bundle_dictionary:
        #             self.__dict__[key] = bundle_dictionary[key]
        # elif not self.allow_partial_reload:
        #     return False
        # else:
        #     tf.logging.warning("Unable to reload the agent's parameters!")
        # Restore the agent's TensorFlow graph.

        self._saver.restore(self.agent.get_sess(),
                            os.path.join(checkpoint_dir,
                                         'tf_ckpt-{}'.format(iteration_number)))
        return True
