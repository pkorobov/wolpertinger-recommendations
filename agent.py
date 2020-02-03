from recsim.agent import AbstractEpisodicRecommenderAgent
import tensorflow as tf
from wolpertinger.wolp_agent import *
from wolpertinger.soft_wolp_agent import *
from stable_baselines.sac import MlpPolicy as SACPolicy
from stable_baselines.ddpg import MlpPolicy as DDPGPolicy
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
                 full_tensorboard_log=False, **kwargs):
        AbstractEpisodicRecommenderAgent.__init__(self, action_space)

        self._observation_space = env.observation_space
        self.agent = WolpertingerAgent(DDPGPolicy, env, tau=1e-5, action_noise=action_noise,
                                       policy_kwargs=policy_kwargs, k_ratio=k_ratio,
                                       full_tensorboard_log=full_tensorboard_log, **kwargs)
        self.t = 0
        self.current_episode = {}
        self.eval_mode = eval_mode
        self.writer = writer

        max_tf_checkpoints_to_keep = 10
        with self.agent.get_sess().as_default(), self.agent.get_graph().as_default():
            self._saver = tf.train.Saver(var_list=tf.all_variables(), max_to_keep=max_tf_checkpoints_to_keep)

    def begin_episode(self, observation=None):
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
        self._saver.save(
            self.agent.get_sess(),
            os.path.join(checkpoint_dir, 'tf_ckpt'),
            global_step=iteration_number)
        bundle_dictionary = {}
        bundle_dictionary['episode_num'] = self._episode_num
        return bundle_dictionary

    def unbundle(self, checkpoint_dir, iteration_number, bundle_dictionary):
        self._saver.restore(self.agent.get_sess(),
                            os.path.join(checkpoint_dir,
                                         'tf_ckpt-{}'.format(iteration_number)))
        return True


class SoftWolpAgent(AbstractEpisodicRecommenderAgent):

    def __init__(self, env, action_space, k_ratio=0.1, policy_kwargs=None,
                 action_noise=None, eval_mode=False, max_actions=1000, writer=None,
                 full_tensorboard_log=False, **kwargs):
        AbstractEpisodicRecommenderAgent.__init__(self, action_space)

        self._observation_space = env.observation_space
        self.agent = SoftWolpertingerAgent(SACPolicy, env, tau=1e-5, action_noise=action_noise,
                                       policy_kwargs=policy_kwargs, k_ratio=k_ratio,
                                       full_tensorboard_log=full_tensorboard_log,
                                       learning_rate=3e-4, **kwargs)
        self.t = 0
        self.current_episode = {}
        self.eval_mode = eval_mode
        self.writer = writer

        # temp
        self.learning_rate = 3e-4

        max_tf_checkpoints_to_keep = 10
        with self.agent.get_sess().as_default(), self.agent.get_graph().as_default():
            self._saver = tf.train.Saver(var_list=tf.all_variables(), max_to_keep=max_tf_checkpoints_to_keep)

    def begin_episode(self, observation=None):
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
        action, q_value = self.agent.predict(state)
        self.current_episode = {
            "obs_t": state,
            "action": action,
        }
        return np.where(action)[0]

    def _observe(self, next_state, reward, done):
        if not self.current_episode:
            raise ValueError("Current episode is expected to be non-empty")

        self.current_episode.update({
            "obs_tp1": next_state,
            "reward": reward,
            "done": float(done)
        })
        if not self.eval_mode:
            # obs_t, action, reward, obs_tp1, done
            self.agent.replay_buffer.add(**self.current_episode)
            if self.agent.replay_buffer.can_sample(self.agent.batch_size):
                self.agent._train_step(self.t, self.writer, self.learning_rate)
                self.agent.sess.run(self.agent.target_update_op)
        self.current_episode = {}

    def _extract_state(self, observation):
        user_space = self._observation_space.spaces['user']
        return spaces.flatten(user_space, observation['user'])

    def bundle_and_checkpoint(self, checkpoint_dir, iteration_number):
        if not tf.gfile.Exists(checkpoint_dir):
            return None
        self._saver.save(
            self.agent.get_sess(),
            os.path.join(checkpoint_dir, 'tf_ckpt'),
            global_step=iteration_number)
        bundle_dictionary = {}
        bundle_dictionary['episode_num'] = self._episode_num
        return bundle_dictionary

    def unbundle(self, checkpoint_dir, iteration_number, bundle_dictionary):
        self._saver.restore(self.agent.get_sess(),
                            os.path.join(checkpoint_dir,
                                         'tf_ckpt-{}'.format(iteration_number)))
        return True