from tensorboard.backend.event_processing import event_accumulator
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib as mpl
import os
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--path', default='logs')
args = parser.parse_args()
mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=['orange', 'green', 'red', 'blue', 'gray'])
plt.rcParams['axes.grid'] = True


def _load_run(path):
    event_acc = event_accumulator.EventAccumulator(path)
    event_acc.Reload()
    data = {}

    for tag in sorted(event_acc.Tags()["scalars"]):
        x, y = [], []
        for scalar_event in event_acc.Scalars(tag):
            x.append(scalar_event.step)
            y.append(scalar_event.value)
        data[tag] = (np.asarray(x), np.asarray(y))
    return data


def plot_averaged_runs(logdir=args.path, averaging=True):

    runs = pd.DataFrame()

    agent_dirs = sorted([*filter(lambda x: os.path.isdir(logdir + '/' + x) and x[0] != '.', os.listdir(logdir))])
    agent_dirs = sorted(agent_dirs, key=len)
    fig, ax = plt.subplots(1, len(agent_dirs), figsize=(len(agent_dirs) * 8, 8))
    plt.xlabel('Iterations')
    plt.ylabel('Mean episode reward')
    for i, agent in enumerate(agent_dirs):
        agent_path = logdir + '/' + agent
        if os.path.isdir(agent_path):
            runs_num = len([*filter(lambda x: os.path.isdir(agent_path + '/' + x) and x[0] != '.', os.listdir(agent_path))])
        else:
            continue
        print("Train rewards of %s" % agent)
        for j in tqdm(range(runs_num)):
            cur_run = pd.Series(_load_run(agent_path + "/run_%s/train" % j)['AverageEpisodeRewards'][1])
            runs['run_%s' % j] = cur_run
        if averaging:
            runs = runs.rolling(window=30).mean()
        means = runs.mean(axis=1)
        stds = runs.std(axis=1)
        ax[i].plot(means.index, means, label=agent, alpha=1.0)
        ax[i].fill_between(means.index, means - stds, means + stds, alpha=0.5)
        ax[i].legend()
    plt.setp(ax, ylim=(-5, 30))
    plt.savefig(logdir + '/averaged_agents.png')

if __name__ == '__main__':
    plot_averaged_runs(args.path)