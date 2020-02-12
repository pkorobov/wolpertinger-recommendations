import os
import shutil

from recsim.agents import full_slate_q_agent
from recsim.agents.random_agent import RandomAgent
from recsim.simulator import recsim_gym, environment
from recsim_custom import EvalRunnerCustom, TrainRunnerCustom
from base.ddpg import GaussNoise, CriticNetwork, ActorNetwork
from stable_baselines.common import set_global_seeds

import numpy as np
import random
import os
from environment import *
from agent import WolpAgent, StaticAgent

RUNS = 5
MAX_TRAINING_STEPS = 15
NUM_ITERATIONS = 400
EVAL_EPISODES = 100
os.environ["CUDA_VISIBLE_DEVICES"]="1"


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


def create_wolp_agent_with_ratio(k_ratio=0.1, policy_kwargs=None, action_noise=None, **kwargs):

    def create_wolp_agent(sess, environment, eval_mode, summary_writer=None):
        return WolpAgent(environment, action_space=environment.action_space,
                         k_ratio=k_ratio, action_noise=action_noise, summary_writer=summary_writer,
                         eval_mode=eval_mode, **kwargs)
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
    SEED = 1
    env.seed(SEED)
    noise = GaussNoise(sigma=0.2 * np.ones(DOC_NUM))

    num_actions = lambda actions, k_ratio: max(1, round(actions * k_ratio))

    agents = [
                ('Wolpertinger ' + "(" + str(num_actions(DOC_NUM, 0.01)) + "NN, normal noise)",
                  create_wolp_agent_with_ratio(0.01, critic_constructor=CriticNetwork,
                                               actor_constructor=ActorNetwork, action_dim=DOC_NUM,
                                               state_dim=DOC_NUM, action_noise=noise)),
                ("Optimal", create_good_agent)
    ]

    base_dir = cleanup_dir('logs/')
    for agent_name, create_agent_fun in agents:
        print("Running %s..." % agent_name)
        for run in range(RUNS):
            SEED = run
            os.environ['PYTHONHASHSEED'] = str(SEED)
            set_global_seeds(SEED)

            print("RUN # %s of %s" % (run, RUNS))
            dir = base_dir + agent_name + "/run_" + str(run)

            runner = TrainRunnerCustom(
                base_dir=dir,
                create_agent_fn=create_agent_fun,
                env=env,
                max_training_steps=MAX_TRAINING_STEPS,
                num_iterations=NUM_ITERATIONS,
                episode_log_file='episodes_log_train.csv'
            )
            runner.run_experiment()

if __name__ == "__main__":
    main()
