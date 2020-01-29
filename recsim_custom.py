from recsim.simulator import runner_lib
import pandas as pd

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

    def _log_one_step(self, user_obs, doc_obs, slate, responses, reward,
                      is_terminal, log_df):
        row = [self.episode_num, user_obs[0], slate[0], reward, is_terminal]
        log_df.loc[len(log_df)] = row

    def _run_one_episode(self):
        """Executes a full trajectory of the agent interacting with the environment.

        Returns:
          The number of steps taken and the total reward.
        """
        step_number = 0
        total_reward = 0.

        start_time = time.time()

        log_df = pd.DataFrame(columns=['episode', 'state', 'recommendation', 'reward', 'is terminal'])
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


class TrainRunnerCustom(runner_lib.TrainRunner, RunnerCustom):
    def __init__(self, max_training_steps=250000, num_iterations=100,
               checkpoint_frequency=1, **kwargs):
        runner_lib.TrainRunner.__init__(self, max_training_steps=max_training_steps,
                                        num_iterations=num_iterations,
                                        checkpoint_frequency=checkpoint_frequency, **kwargs)


class EvalRunnerCustom(runner_lib.EvalRunner, RunnerCustom):
    def __init__(self, max_eval_episodes=125000,
                 test_mode=False,
                 min_interval_secs=30,
                 train_base_dir=None,
                 **kwargs):
        runner_lib.EvalRunner.__init__(self, max_eval_episodes=max_eval_episodes,
                                               test_mode=test_mode,
                                               min_interval_secs=min_interval_secs,
                                               train_base_dir=train_base_dir,
                                               **kwargs)

    def _run_eval_phase(self, total_steps):
        """Runs evaluation phase given model has been trained for total_steps."""

        self._env.reset_sampler()
        self._initialize_metrics()

        num_episodes = 0
        episode_rewards = []

        while num_episodes < self._max_eval_episodes:
            self._initialize_metrics()
            _, episode_reward = self._run_one_episode()
            episode_rewards.append(episode_reward)
            num_episodes += 1
            self._write_metrics(num_episodes, suffix='eval')

        output_file = os.path.join(self._output_dir, 'returns_%s' % total_steps)
        tf.compat.v1.logging.info('eval_file: %s', output_file)
        with tf.io.gfile.GFile(output_file, 'w+') as f:
            f.write(str(episode_rewards))

