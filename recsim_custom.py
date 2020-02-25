from recsim.simulator import runner_lib
import recsim

import pandas as pd
from tensorboardX import SummaryWriter
import environment
from environment import DOC_NUM
from base.ddpg import GaussNoise

import numpy as np

import os
import time

from gym import spaces
import numpy as np
import tensorflow as tf

class RunnerCustom(runner_lib.Runner):

    def __init__(self,
                 base_dir,
                 create_agent_fn,
                 env,
                 episode_log_file='',
                 checkpoint_file_prefix='ckpt',
                 max_steps_per_episode=27000):

        tf.compat.v1.logging.info('max_steps_per_episode = %s',
                                  max_steps_per_episode)

        if base_dir is None:
            raise ValueError('Missing base_dir.')

        self.episode_num = 0
        self._base_dir = base_dir
        self._create_agent_fn = create_agent_fn
        self._env = env
        self._checkpoint_file_prefix = checkpoint_file_prefix
        self._max_steps_per_episode = max_steps_per_episode
        self._episode_log_path = os.path.join(base_dir, episode_log_file) if episode_log_file else None
        self._episode_log_file = None  # for the sake of turning off bloody TFWriter
        self._episode_writer = None

    def _set_up(self, eval_mode):
        self._summary_writer = SummaryWriter(self._output_dir)
        self._sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        self._agent = self._create_agent_fn(
            self._sess,
            self._env,
            summary_writer=self._summary_writer,
            eval_mode=eval_mode)
        # type check: env/agent must both be multi- or single-user
        if self._agent.multi_user and not isinstance(
                self._env.environment, recsim.simulator.environment.MultiUserEnvironment):
            raise ValueError('Multi-user agent requires multi-user environment.')
        if not self._agent.multi_user and isinstance(
                self._env.environment, recsim.simulator.environment.MultiUserEnvironment):
            raise ValueError('Single-user agent requires single-user environment.')

    def _log_one_step(self, user_obs, doc_obs, slate, responses, reward,
                      is_terminal, log_df):
        row = [self.episode_num, user_obs[0], slate[0], reward]
        log_df.loc[len(log_df)] = row

    def _run_one_episode(self):
        """Executes a full trajectory of the agent interacting with the environment.

        Returns:
          The number of steps taken and the total reward.
        """
        step_number = 0
        total_reward = 0.

        start_time = time.time()

        log_df = pd.DataFrame(columns=['episode', 'state', 'recommendation', 'reward'])
        observation = self._env.reset()
        action = self._agent.begin_episode(observation)

        # Keep interacting until we reach a terminal state.
        while True:
            last_observation = observation
            observation, reward, done, info = self._env.step(action)
            self._log_one_step(last_observation['user'], last_observation['doc'],
                               action, observation['response'], reward, done,
                               log_df)
            # Update environment-specific metrics with responses to the slate.
            self._env.update_metrics(observation['response'], info)

            total_reward += reward
            step_number += 1

            if done:
                break
            elif step_number == self._max_steps_per_episode:
                # Stop the run loop once we reach the true end of episode.
                break
            else:
                action = self._agent.step(reward, observation)

        self._agent.end_episode(reward, observation)
        if self._episode_log_path is not None:
            if os.path.exists(self._episode_log_path):
                log_df.to_csv(self._episode_log_path, mode='a', header=False, index=False)
            else:
                log_df.to_csv(self._episode_log_path, header=True, index=False)

        time_diff = time.time() - start_time
        self._update_episode_metrics(
            episode_length=step_number,
            episode_time=time_diff,
            episode_reward=total_reward)

        self.episode_num += 1
        return step_number, total_reward


    def _write_metrics(self, step, suffix):
        """Writes the metrics to Tensorboard summaries."""
        num_steps = np.sum(self._stats['episode_length'])
        time_per_step = np.sum(self._stats['episode_time']) / num_steps
        self._summary_writer.add_scalar('AverageEpisodeRewards', np.mean(self._stats['episode_reward']), step)
        # self._summary_writer.add_scalar('AverageEpisodeLength', np.mean(self._stats['episode_length']), step)
        self._summary_writer.flush()


class TrainRunnerCustom(runner_lib.TrainRunner, RunnerCustom):
    def __init__(self, max_training_steps=250000, num_iterations=100,
                 checkpoint_frequency=1, experiment_type="dynamic_change",
                 change_freq=100, seed=1, **kwargs):
        runner_lib.TrainRunner.__init__(self, max_training_steps=max_training_steps,
                                        num_iterations=num_iterations,
                                        checkpoint_frequency=checkpoint_frequency,
                                        **kwargs)
        self.experiment_type = experiment_type
        self.change_freq = change_freq

        np.random.seed(seed)

        global SECOND_POPULAR, FIRST_POPULAR
        if self.experiment_type == "alternating_most_acceptable" or \
           self.experiment_type == "static_dominant":
            environment.W = np.random.uniform(0.0, 0.2, (DOC_NUM, DOC_NUM))
            environment.MOST_POPULAR = 6
            environment.W[:, environment.MOST_POPULAR] = np.random.uniform(0.9, 1.0, DOC_NUM)
        if self.experiment_type == "alternating_pair":
            environment.W = np.random.uniform(0.0, 0.2, (DOC_NUM, DOC_NUM))
            FIRST_POPULAR = 6
            SECOND_POPULAR = 47
            environment.W[:, FIRST_POPULAR] = np.random.uniform(0.8, 0.9, DOC_NUM)
            environment.W[:, SECOND_POPULAR] = np.random.uniform(0.6, 0.7, DOC_NUM)
            environment.W[FIRST_POPULAR, FIRST_POPULAR] = np.random.uniform(0.4, 0.5)
            environment.W[SECOND_POPULAR, SECOND_POPULAR] = np.random.uniform(0.2, 0.3)
            environment.W[FIRST_POPULAR, SECOND_POPULAR] = np.random.uniform(0.9, 1.0)
        if self.experiment_type == "shift":
            environment.W = np.diag(np.random.uniform(0.8, 0.9, DOC_NUM))
            environment.W += (np.ones((DOC_NUM, DOC_NUM)) - np.eye(DOC_NUM)) * np.random.uniform(0.0, 0.2, (DOC_NUM, DOC_NUM))
            environment.W = np.roll(environment.W, 1, axis=1)
            print(environment.W)

    def run_experiment(self):
        """Runs a full experiment, spread over multiple iterations."""
        tf.logging.info('Beginning training...')
        start_iter, total_steps = self._initialize_checkpointer_and_maybe_resume(
            self._checkpoint_file_prefix)
        if self._num_iterations <= start_iter:
            tf.logging.warning('num_iterations (%d) < start_iteration(%d)',
                               self._num_iterations, start_iter)
            return

        for iteration in range(start_iter, self._num_iterations):

            tf.logging.info('Starting iteration %d', iteration)
            total_steps = self._run_train_phase(total_steps)
            if iteration % self._checkpoint_frequency == 0:
                self._checkpoint_experiment(iteration, total_steps)
            if iteration == self.change_freq + 100 and self.experiment_type == "alternating_most_acceptable":
                if hasattr(self._agent, 'agent'):
                    self._agent.agent.noise = GaussNoise(sigma=0.05 * np.ones(DOC_NUM))
            if iteration % self.change_freq == 0 and iteration > 0:
                self._update_w()
                if hasattr(self._agent, 'agent') and self.experiment_type == "alternating_most_acceptable":
                    self._agent.agent.noise = GaussNoise(sigma=1.0 * np.ones(DOC_NUM))

    def _run_train_phase(self, total_steps):
        """Runs training phase and updates total_steps."""

        self._initialize_metrics()

        num_steps = 0
        while num_steps < self._max_training_steps:
            episode_length, _ = self._run_one_episode()
            num_steps += episode_length

        total_steps += num_steps
        self._write_metrics(total_steps, suffix='train')
        return total_steps

    def _update_w(self):
        global SECOND_POPULAR, FIRST_POPULAR, MOST_POPULAR, W
        if self.experiment_type == "alternating_most_acceptable":
            environment.W = np.random.uniform(0.0, 0.2, (DOC_NUM, DOC_NUM))
            new_most_popular = np.random.randint(DOC_NUM)
            while new_most_popular == environment.MOST_POPULAR:
                new_most_popular = np.random.randint(DOC_NUM)
            else:
                environment.MOST_POPULAR = new_most_popular
            environment.W[:, environment.MOST_POPULAR] = np.random.uniform(0.9, 1.0, DOC_NUM)
        if self.experiment_type == "alternating_pair":
            environment.W = np.random.uniform(0.0, 0.2, (DOC_NUM, DOC_NUM))
            SECOND_POPULAR, FIRST_POPULAR = FIRST_POPULAR, SECOND_POPULAR
            environment.W[:, FIRST_POPULAR] = np.random.uniform(0.8, 0.9, DOC_NUM)
            environment.W[:, SECOND_POPULAR] = np.random.uniform(0.6, 0.7, DOC_NUM)
            environment.W[FIRST_POPULAR, FIRST_POPULAR] = np.random.uniform(0.4, 0.5)
            environment.W[SECOND_POPULAR, SECOND_POPULAR] = np.random.uniform(0.2, 0.3)
            environment.W[FIRST_POPULAR, SECOND_POPULAR] = np.random.uniform(0.9, 1.0)
