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
import plots
import plotly.graph_objects as go
import plotly
import pandas as pd

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
              seed, max_total_steps, times_to_evaluate, eval_mode=False):

    log_df = pd.DataFrame(columns=['episode', 's', 'a', 'opt a', 'reward'])

    eval_freq = max_total_steps // times_to_evaluate
    eval_q_table_freq = max_total_steps // 10

    summary_writer = SummaryWriter(base_dir / f"{agent_name}/run_{seed}/train")
    fix_seed(seed)
    c.init_w()

    agent = create_function(None, env, eval_mode=eval_mode, summary_writer=summary_writer)

    if type(agent) == WolpertingerRecSim:
        heatmaps = {"q_values": go.Figure(),
                    "q_values_target": go.Figure(),
                    "policy": go.Figure(),
                    "W": go.Figure()}

    step_number = 0
    cum_reward = 0
    episode_number = 0
    episodes_to_avg = 0
    observation = env.reset()

    while step_number < max_total_steps:

        episode_reward = 0
        action = agent.begin_episode(observation)
        episode_len = 0
        while True:
            observation, reward, done, info = env.step(action)
            step_number += 1
            episode_len += 1
            episode_reward += reward
            if done:
                break
            else:
                action = agent.step(reward, observation)

            s = observation['user'].item()
            a = np.array(action).item()
            a_opt = c.OPTIMAL_ACTIONS[s]

            log_df.loc[step_number] = [episode_number, s, a, a_opt, reward]
        episode_number += 1

        agent.end_episode(reward, observation)
        cum_reward += episode_reward
        episodes_to_avg += 1

        if type(agent) == WolpertingerRecSim and step_number // eval_q_table_freq != (step_number - episode_len) // eval_q_table_freq:
            q_values = np.hstack([agent._agent.compute_q_values(i) for i in range(c.DOC_NUM)]).T
            q_values_target = np.hstack([agent._agent.compute_q_values(i, target=True) for i in range(c.DOC_NUM)]).T
            actions = np.vstack([agent._agent
                                 .proto_action(c.EMBEDDINGS[i], with_noise=False)
                                 for i in range(c.DOC_NUM)])
            heatmaps["q_values"].add_trace(go.Heatmap(z=q_values))
            heatmaps["q_values_target"].add_trace(go.Heatmap(z=q_values_target))
            heatmaps["policy"].add_trace(go.Heatmap(z=actions))

        if step_number // eval_freq != (step_number - episode_len) // eval_freq:
            print(step_number, cum_reward / episodes_to_avg)
            summary_writer.add_scalar('AverageEpisodeRewards', cum_reward / episodes_to_avg, step_number)
            episodes_to_avg = 0
            cum_reward = 0

    if type(agent) == WolpertingerRecSim:
        heatmaps["W"].add_trace(go.Heatmap(z=c.W))
        for key, fig in heatmaps.items():
            plots.plotly_heatmap(fig, base_dir / agent_name / f"run_{seed}/{key}.html")
        agent._agent.save(str(base_dir / agent_name / f"run_{seed}/parameters"))

    log_df.to_csv(base_dir / agent_name / f"run_{seed}/steps.csv")
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
    parser.add_argument('--times_to_evaluate', type=int, default=1000)
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
            logging.info(f"RUN #{run + 1} of {args.runs} ({agent_name})")
            run_agent(env, create_function, agent_name, base_dir, run,
                      args.total_steps, args.times_to_evaluate, eval_mode=False)

    logging.disable()
    plot_averaged_runs(str(base_dir), ylimits=[0, 12], smoothing=False)


if __name__ == "__main__":
    main()
