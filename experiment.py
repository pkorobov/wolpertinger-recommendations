import os
import shutil

from recsim.agents import full_slate_q_agent
from recsim.agents.random_agent import RandomAgent
from recsim.simulator import recsim_gym, environment
from recsim_custom import TrainRunnerCustom
from base.ddpg import GaussNoise
from plots import plot_averaged_runs

import torch
import numpy as np
import random
import os
from environment import *
from agent import WolpAgent, StaticAgent
from datetime import datetime
import json

RUNS = 5
MAX_TRAINING_STEPS = 15
NUM_ITERATIONS = 3000
EVAL_EPISODES = 100

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

start_time = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')


def create_random_agent(sess, environment, eval_mode, summary_writer=None):
    return RandomAgent(environment.action_space, random_seed=SEED)


def create_good_agent(sess, environment, eval_mode, summary_writer=None):
    return StaticAgent(environment, 6)


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

    # static env
    # parameters = {'action_dim': DOC_NUM,
    #               'state_dim': DOC_NUM,
    #               'noise': GaussNoise(sigma=0.1),
    #               'critic_lr': 1e-3,
    #               'actor_lr': 1e-4,
    #               'soft_tau': 1e-3,
    #               'hidden_dim': 16,
    #               'batch_size': 128,
    #               'buffer_size': 1000,
    #               'gamma': 0.99}

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
                  'actor_weight_decay': 0.001,
                  'critic_weight_decay': 0.001,
                  'eps': 1e-1
                  # 'training_starts': 1000
                  }

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

    num_actions = lambda actions, k_ratio: max(1, int(actions * k_ratio))
    agents = [
                # ('Wolpertinger ' + "(" + str(num_actions(DOC_NUM, 0.01)) + "NN, normal noise)",
                #  create_wolp_agent_with_ratio(0.01, **parameters)),
                ('Wolpertinger ' + "(" + str(num_actions(DOC_NUM, 0.33)) + "NN, normal noise)",
                 create_wolp_agent_with_ratio(0.33, **parameters)),
                # ('Wolpertinger ' + "(" + str(num_actions(DOC_NUM, 1.0)) + "NN, normal noise)",
                #  create_wolp_agent_with_ratio(1.0, **parameters)),
                ("Optimal", create_good_agent),
    ]

    # base_dir = cleanup_dir('logs/')

    # experiment_type = 'static_dominant'
    # experiment_type = 'alternating_most_acceptable'
    # experiment_type = 'alternating_pair'
    experiment_type = 'shift'

    base_dir = 'logs/' + experiment_type + ' (' + start_time + ')'
    os.makedirs(base_dir)

    # with open(base_dir + 'params.txt', 'w') as f:
    #     f.write(json.dumps(parameters))

    for agent_name, create_agent_fun in agents:
        print("Running %s..." % agent_name)
        for run in range(RUNS):
            SEED = run
            os.environ['PYTHONHASHSEED'] = str(SEED)
            np.random.seed(SEED)
            torch.manual_seed(SEED)

            print("RUN # %s of %s" % (run, RUNS))
            dir = base_dir + '/' + agent_name + "/run_" + str(run)

            runner = TrainRunnerCustom(
                base_dir=dir,
                create_agent_fn=create_agent_fun,
                env=env,
                max_training_steps=MAX_TRAINING_STEPS,
                num_iterations=NUM_ITERATIONS,
                episode_log_file='episodes_log_train.csv',
                experiment_type=experiment_type,
                seed=SEED,
                change_freq=400
            )
            runner.run_experiment()
    plot_averaged_runs(base_dir)

if __name__ == "__main__":
    main()
