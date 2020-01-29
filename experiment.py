import os
import shutil

from recsim.agents import full_slate_q_agent
from recsim.agents.random_agent import RandomAgent
from recsim.simulator import recsim_gym, environment
from recsim_custom import EvalRunnerCustom, TrainRunnerCustom
from stable_baselines.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec
import tensorflow as tf

from environment import *
from agent import WolpAgent, StaticAgent

RUNS = 1
MAX_TRAINING_STEPS = 15
NUM_ITERATIONS = 400

def create_random_agent(sess, environment, eval_mode, summary_writer=None):
    return RandomAgent(environment.action_space, random_seed=SEED)


def create_good_agent(sess, environment, eval_mode, summary_writer=None):
    return StaticAgent(environment.action_space, 6)


def create_bad_agent(sess, environment, eval_mode, summary_writer=None):
    return StaticAgent(environment.action_space, 0)


def create_dqn_agent(sess, environment, eval_mode, summary_writer=None):
    kwargs = {
        'observation_space': environment.observation_space,
        'action_space': environment.action_space,
        'summary_writer': summary_writer,
        'eval_mode': eval_mode,
    }
    return full_slate_q_agent.FullSlateQAgent(sess, **kwargs)


def create_wolp_agent_with_ratio(k_ratio=0.1, policy_kwargs=None, action_noise=None):

    def create_wolp_agent(sess, environment, eval_mode, summary_writer=None):
        # action_noise_ = None if eval_mode else action_noise
        return WolpAgent(environment, action_space=environment.action_space,
                         k_ratio=k_ratio, policy_kwargs=policy_kwargs,
                         action_noise=action_noise, eval_mode=eval_mode)

    return create_wolp_agent


def cleanup_dir(dir_path):
    if os.path.exists(dir_path) and os.path.isdir(dir_path):
        print("Cleaning up {}".format(dir_path))
        shutil.rmtree(dir_path)
    os.makedirs(dir_path)
    return dir_path

def main():
    """
    See results with to compare different agents
      tensorboard --logdir /tmp/recsim
    """
    env = recsim_gym.RecSimGymEnv(
        environment.Environment(UserModel(), DocumentSampler(), DOC_NUM, 1, resample_documents=False),
        clicked_reward
    )

    policy_kwargs = {'layers': [64, 64]}
    # noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(DOC_NUM), sigma=0.3 * np.ones(DOC_NUM))
    noise = NormalActionNoise(mean=np.zeros(DOC_NUM), sigma=0.2 * np.ones(DOC_NUM))

    num_actions = lambda actions, k_ratio: max(1, round(actions * k_ratio))

    agents = [
               ("random", create_random_agent),
               ("optimal", create_good_agent),
               ("bad", create_bad_agent),
               # ('wolpertinger_1_neighbour', create_wolp_agent_with_ratio(0, action_noise=OrnsteinUhlenbeckActionNoise)),
               ('wolpertinger_0.01_' + "(" + str(num_actions(DOC_NUM, 0.01)) + ")",
                                                                  create_wolp_agent_with_ratio(0.01,
                                                                  action_noise=noise,
                                                                  policy_kwargs=policy_kwargs)),
               # ('wolpertinger_0.05_' + "(" + str(num_actions(DOC_NUM, 0.05)) + ")",
               #                                                    create_wolp_agent_with_ratio(0.05,
               #                                                    action_noise=noise,
               #                                                    policy_kwargs=policy_kwargs)),
               ('wolpertinger_0.1_' + "(" + str(num_actions(DOC_NUM, 0.1)) + ")",
                                                                 create_wolp_agent_with_ratio(0.1,
                                                                 action_noise=noise,
                                                                 policy_kwargs=policy_kwargs)),
               ('wolpertinger_1.0_' + "(" + str(num_actions(DOC_NUM, 1.0)) + ")",
                                                                 create_wolp_agent_with_ratio(1,
                                                                 action_noise=noise,
                                                                 policy_kwargs=policy_kwargs))
    ]

    base_dir = cleanup_dir('logs/')
    for agent_name, create_agent_fun in agents:
        for run in range(RUNS):
            dir = base_dir + agent_name + "/run_" + str(run)
            env.reset()

            runner = TrainRunnerCustom(
                base_dir=dir,
                create_agent_fn=create_agent_fun,
                env=env,
                max_training_steps=MAX_TRAINING_STEPS,
                num_iterations=NUM_ITERATIONS,
                episode_log_file='episodes_log_train.csv'
            )
            runner.run_experiment()

            runner = EvalRunnerCustom(
                base_dir=dir,
                create_agent_fn=create_agent_fun,
                env=env,
                max_eval_episodes=1000,
                test_mode=True,
                episode_log_file='episodes_log_eval.csv'
            )
            runner.run_experiment()

if __name__ == "__main__":
    main()
