import shutil
import torch
from recsim.simulator import recsim_gym, environment
from plots import plot_averaged_runs
from tensorboardX import SummaryWriter
import os
from datetime import datetime
import random
import argparse
import config as c
import matrix_env as me
from agent import WolpertingerRecSim, OptimalAgent
from recsim.agents.random_agent import RandomAgent
import numpy as np
import itertools
from pathlib import Path
import logging

parser = argparse.ArgumentParser()
parser.add_argument('--parameters', default='parameters.json')
args = parser.parse_args()
c.init_config(args.parameters)


RUNS = 5
MAX_TOTAL_STEPS = c.MAX_TOTAL_STEPS

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def setup_logging():
    logging.basicConfig(format='[%(asctime)s] %(levelname)s %(message)s', level=logging.INFO, datefmt='%I:%M:%S')
    logging.getLogger().setLevel(logging.INFO)


def create_random_agent(sess, env, **kwargs):
    return RandomAgent(env.action_space)


def create_optimal_agent(sess, env, **kwargs):
    return OptimalAgent(env)


def create_wolp_agent_with_ratio(k_ratio=0.1, **kwargs):
    def create_wolp_agent(sess, env, eval_mode, summary_writer=None):
        return WolpertingerRecSim(env, action_space=env.action_space,
                                  k_ratio=k_ratio, summary_writer=summary_writer,
                                  eval_mode=eval_mode, **kwargs)

    return create_wolp_agent


start_time = datetime.now().strftime('%y.%m.%d %H-%M')


def cleanup_dir(dir_path):
    if os.path.exists(dir_path) and os.path.isdir(dir_path):
        print("Cleaning up {}".format(dir_path))
        shutil.rmtree(dir_path)
    os.makedirs(dir_path)
    return dir_path


def fix_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)


def main():
    """
    See results with to compare different agents
      tensorboard --logdir logs --samples_per_plugin "images=100"
    """
    setup_logging()

    env = recsim_gym.RecSimGymEnv(
        environment.SingleUserEnvironment(
                        me.UserModel(), me.DocumentSampler(), c.DOC_NUM,
                        slate_size=1, resample_documents=False),
        me.clicked_reward
    )

    # base_dir = 'logs/' + c.ENV_PARAMETERS['type'] + '/' + start_time + '/'
    base_dir = Path('logs') / start_time / c.ENV_PARAMETERS['type']
    # os.makedirs(base_dir, exist_ok=True)

    # base_dir = cleanup_dir(Path('logs') / c.ENV_PARAMETERS['type'])

    def wolpertinger_name(actions, k_ratio, param_string):
        k = max(1, int(actions * k_ratio))
        return "Wolpertinger {}NN ({})".format(k, param_string)

    k_ratios = [0.33]

    agents = [
        ("Optimal", create_optimal_agent)
    ]

    dim = c.EMBEDDINGS.shape[1]
    for k_ratio, (parameters, param_string) in itertools.product(k_ratios, zip(c.AGENT_PARAMETERS, c.AGENT_PARAM_STRINGS)):
        agents.append(
                (wolpertinger_name(c.DOC_NUM, k_ratio, param_string),
                 create_wolp_agent_with_ratio(k_ratio, state_dim=dim, action_dim=dim,
                                              embeddings=c.EMBEDDINGS, **parameters))
        )

    for agent_number, (agent_name, create_function) in enumerate(agents):
        logging.info(f"Running agent #{agent_number + 1} of {len(agents)}...")
        for run in range(RUNS):

            logging.info(f"RUN #{run + 1} of {RUNS}")
            summary_writer = SummaryWriter(base_dir / f"{agent_name}/run_{run}/train")
            fix_seed(run)
            c.init_w()

            agent = create_function(None, env, eval_mode=False, summary_writer=summary_writer)
            step_number = 0
            observation = env.reset()
            while step_number < MAX_TOTAL_STEPS:
                episode_reward = 0

                # episode
                action = agent.begin_episode(observation)
                while True:
                    observation, reward, done, info = env.step(action)
                    step_number += 1
                    episode_reward += reward
                    if done:
                        break
                    else:
                        action = agent.step(reward, observation)
                agent.end_episode(reward, observation)

                summary_writer.add_scalar('AverageEpisodeRewards', episode_reward, step_number)
            summary_writer.close()

    logging.disable()
    plot_averaged_runs(str(base_dir))


if __name__ == "__main__":
    main()
