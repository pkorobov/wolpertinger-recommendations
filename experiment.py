import torch
from recsim.simulator import recsim_gym, environment
from tensorboardX import SummaryWriter
import os
import random
import argparse
import config as c
import matrix_env as me
from agent import WolpertingerRecSim, OptimalAgent
from recsim.agents.random_agent import RandomAgent
import numpy as np
from pathlib import Path
from functools import partial
import plots
import plotly.graph_objects as go
import pandas as pd

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def create_random_agent(sess, env, **kwargs):
    return RandomAgent(env.action_space)


def create_optimal_agent(sess, env, **kwargs):
    return OptimalAgent(env)


def create_wolp_agent(sess, env, eval_mode, k_ratio=0.1, summary_writer=None, **kwargs):
    return WolpertingerRecSim(env, k_ratio=k_ratio, summary_writer=summary_writer,
                              eval_mode=eval_mode, **kwargs)


def fix_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def run_agent(env, create_function, agent_name, base_dir,
              seed, max_total_steps, times_to_evaluate,
              eval_mode=False, display=False):

    rewards = []
    log_df = pd.DataFrame(columns=['episode', 's', 'a', 'opt a', 'a_proba', 'opt_proba', 'reward'])

    eval_freq = max(max_total_steps // times_to_evaluate, 1)
    eval_q_table_freq = max(max_total_steps // 10, 1)

    summary_writer = SummaryWriter(base_dir / f"{agent_name}" / f"run_{seed}/train")
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
            rewards.append(reward)

            if done:
                break
            else:
                action = agent.step(reward, observation)

            s = observation['user'].item()
            a = np.array(action).item()
            a_opt = c.OPTIMAL_ACTIONS[s]
            a_proba = c.W[s, a].round(3)
            opt_proba = c.OPTIMAL_PROBAS[s].round(3)
            log_df.loc[step_number] = [episode_number, s, a, a_opt, a_proba, opt_proba, reward]
        episode_number += 1

        agent.end_episode(reward, observation)
        cum_reward += episode_reward
        episodes_to_avg += 1

        if type(agent) == WolpertingerRecSim and step_number // eval_q_table_freq != (step_number - episode_len) // eval_q_table_freq:
            q_values = np.hstack([agent.core_agent.compute_q_values(i) for i in range(c.DOC_NUM)]).T
            q_values_target = np.hstack([agent.core_agent.compute_q_values(i, target=True) for i in range(c.DOC_NUM)]).T
            actions = np.vstack([agent.core_agent
                                 .proto_action(c.EMBEDDINGS[i], with_noise=False)
                                 for i in range(c.DOC_NUM)])
            heatmaps["q_values"].add_trace(go.Heatmap(z=q_values))
            heatmaps["q_values_target"].add_trace(go.Heatmap(z=q_values_target))
            heatmaps["policy"].add_trace(go.Heatmap(z=actions))

            os.makedirs(str(base_dir / agent_name / f"run_{seed}/parameters"), exist_ok=True)
            agent.core_agent.save(str(base_dir / agent_name / f"run_{seed}/parameters/{step_number // eval_q_table_freq}"))
            log_df.to_csv(base_dir / agent_name / f"run_{seed}/steps.csv")

        if step_number // eval_freq != (step_number - episode_len) // eval_freq:
            if display:
                print(step_number, cum_reward / episodes_to_avg)
            summary_writer.add_scalar('AverageEpisodeRewards', cum_reward / episodes_to_avg, step_number)
            episodes_to_avg = 0
            cum_reward = 0

    if type(agent) == WolpertingerRecSim:
        heatmaps["W"].add_trace(go.Heatmap(z=c.W))
        for key, fig in heatmaps.items():
            plots.plotly_heatmap(fig, base_dir / agent_name / f"run_{seed}/{key}.html")
        agent.core_agent.save(str(base_dir / agent_name / f"run_{seed}/parameters"))

    np.save(base_dir / agent_name / f"run_{seed}/rewards.npy", np.array(rewards))
    log_df.to_csv(base_dir / agent_name / f"run_{seed}/steps.csv")
    summary_writer.close()


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--parameters', default='parameters.json')
    parser.add_argument('--total_steps', type=int, default=10**5)
    parser.add_argument('--times_to_evaluate', type=int, default=1000)
    parser.add_argument('--logdir', default='logs')
    parser.add_argument('--rmdir', type=bool, default=False)
    parser.add_argument('--display', type=bool, default=False)
    parser.add_argument('--output_path', type=str, default=None)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--agent', type=str, default='Wolpertinger')
    parser.add_argument('--param_string', type=str, default="")

    args = parser.parse_args()
    c.init_config(args.parameters)

    env = recsim_gym.RecSimGymEnv(
        environment.SingleUserEnvironment(
                        me.UserModel(), me.DocumentSampler(), c.DOC_NUM,
                        slate_size=1, resample_documents=False),
        me.clicked_reward
    )

    base_dir = Path(args.logdir) / c.ENV_PARAMETERS['type']

    def wolpertinger_name(actions, k_ratio, param_string):
        k = max(1, int(actions * k_ratio))
        return f"Wolpertinger {k}NN" + (f" ({param_string})" if param_string else "")

    dim = c.EMBEDDINGS.shape[1]
    agent_name = args.agent
    if agent_name == "Wolpertinger":
        k_ratio = 0.0
        if "k_ratio" in c.AGENT_PARAMETERS:
            k_ratio = c.AGENT_PARAMETERS.pop("k_ratio")
        create_function = partial(create_wolp_agent, k_ratio=k_ratio, state_dim=dim, action_dim=dim,
                                  embeddings=c.EMBEDDINGS, **c.AGENT_PARAMETERS)
        agent_name = wolpertinger_name(c.DOC_NUM, k_ratio, args.param_string)

    elif agent_name == "Random":
        create_function = create_random_agent

    else:
        create_function = create_optimal_agent

    run_agent(env, create_function, agent_name, base_dir, args.seed,
              args.total_steps, args.times_to_evaluate,
              eval_mode=False, display=args.display)


if __name__ == "__main__":
    main()
