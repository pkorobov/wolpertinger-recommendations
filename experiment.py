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
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
from base.ddpg import DDPG
from base.td3 import TD3
import gym

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


def create_wolp_agent(sess, env, eval_mode, k_ratio=0.1, backend=DDPG, summary_writer=None, **kwargs):
    if type(backend) == str:
        backend = eval(backend)
    return WolpertingerRecSim(env, k_ratio=k_ratio, summary_writer=summary_writer,
                              eval_mode=eval_mode, backend=backend, **kwargs)


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


def run_agent(env, create_function, agent_name, base_dir,
              seed, max_total_steps, write_metrics_freq, eval_mode=False):

    summary_writer = SummaryWriter(base_dir / f"{agent_name}/run_{seed}/train")
    fix_seed(seed)
    c.init_w()

    agent = create_function(None, env, eval_mode=eval_mode, summary_writer=summary_writer)
    step_number = 0
    episode_number = 0
    cum_reward = 0
    observation = env.reset()

    while step_number < max_total_steps:

        episode_reward = 0
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
        episode_number += 1
        cum_reward += episode_reward

        if episode_number % write_metrics_freq == 0:
            print(step_number, cum_reward / write_metrics_freq)
            summary_writer.add_scalar('AverageEpisodeRewards', cum_reward / write_metrics_freq, step_number)
            cum_reward = 0

    summary_writer.close()


def main():
    """
    See results with to compare different agents
      tensorboard --logdir logs --samples_per_plugin "images=100"
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--parameters', default='parameters.json')
    parser.add_argument('--runs', type=int, default=5)
    parser.add_argument('--total_steps', type=int, default=10**5)
    parser.add_argument('--write_metrics_freq', type=int, default=1)
    parser.add_argument('--logdir', default='logs')
    parser.add_argument('--rmdir', type=bool, default=False)

    args = parser.parse_args()
    c.init_config(args.parameters)

    setup_logging()

    env = recsim_gym.RecSimGymEnv(
        environment.SingleUserEnvironment(
                        me.UserModel(), me.DocumentSampler(), c.DOC_NUM,
                        slate_size=1, resample_documents=False),
        me.clicked_reward
    )

    base_dir = Path(args.logdir) / c.ENV_PARAMETERS['type']
    if args.rmdir:
        cleanup_dir(base_dir)

    def wolpertinger_name(actions, k_ratio, param_string):
        k = max(1, int(actions * k_ratio))
        return "Wolpertinger {}NN ({})".format(k, param_string)

    k_ratios = [0.1]

    agents = []
    dim = c.EMBEDDINGS.shape[1]
    for k_ratio, (parameters, param_string) in itertools.product(k_ratios, zip(c.AGENT_PARAMETERS, c.AGENT_PARAM_STRINGS)):

        agent = parameters.pop("agent")
        if agent == 'Wolpertinger':
            create_function = partial(create_wolp_agent, k_ratio=k_ratio, state_dim=dim, action_dim=dim,
                                      embeddings=c.EMBEDDINGS, **parameters)
            agents.append(
                    (wolpertinger_name(c.DOC_NUM, k_ratio, param_string),
                     create_function)
            )

        if agent == 'Random':
            agents.append(("Random", create_random_agent))

        if agent == 'Optimal':
            agents.append(("Optimal", create_optimal_agent))

    for agent_number, (agent_name, create_function) in enumerate(agents):
        logging.info(f"Running agent #{agent_number + 1} of {len(agents)}...")

        for run in range(args.runs):
            logging.info(f"RUN #{run + 1} of {args.runs}")
            run_agent(env, create_function, agent_name, base_dir, run,
                      args.total_steps, args.write_metrics_freq, eval_mode=False)

    logging.disable()
    plot_averaged_runs(str(base_dir), ylimits=[0, 12])


if __name__ == "__main__":
    main()
