from tensorboard.backend.event_processing import event_accumulator
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib as mpl
import os
from tqdm import tqdm
import argparse
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

parser = argparse.ArgumentParser()
parser.add_argument('--path', default='logs')
args = parser.parse_args()
mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=['orange', 'green', 'red', 'blue', 'gray'])
plt.rcParams['axes.grid'] = True


def heatmap(summary_writer, matrix, tag, step):
    fig, ax = plt.subplots(figsize=matrix.shape)
    sns.heatmap(matrix, annot=True)
    summary_writer.add_figure(tag, fig, global_step=step)
    return fig


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
        if len(agent_dirs) == 1:
            ax.plot(means.index, means, label=agent, alpha=1.0)
            ax.fill_between(means.index, means - stds, means + stds, alpha=0.5)
            ax.legend(fontsize=8)
        else:
            ax[i].plot(means.index, means, label=agent, alpha=1.0)
            ax[i].fill_between(means.index, means - stds, means + stds, alpha=0.5)
            ax[i].legend(fontsize=8)

    plt.setp(ax, ylim=(-5, 30))
    plt.savefig(logdir + '/averaged_agents.png')


def plot_2d_function(summary_writer, func, tag, step):
    x = np.arange(0.0, 1.0, 0.01)
    y = np.arange(0.0, 1.0, 0.01)
    X, Y = np.meshgrid(x, y)  # grid of point
    Z = np.zeros(X.shape)
    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            Z[i, j] = func(X[i, j], X[i, j])
    fig = plt.figure()

    ax = fig.gca(projection='3d')
    ax.scatter(X[0, -1], Y[0, -1], Z[0, -1], c='green')
    ax.scatter(X[-1, 0], Y[-1, 0], Z[-1, 0], c='red')
    ax.scatter(X[0, 0], Y[0, 0], Z[0, 0], c='gray')
    ax.scatter(X[-1, -1], Y[-1, -1], Z[-1, -1], c='gray')

    ax.text(X[0, -1], Y[0, -1], Z[0, -1] + 0.05, str(Z[0, -1].round(5)), c='green')
    ax.text(X[-1, 0], Y[-1, 0], Z[-1, 0] + 0.1, str(Z[-1, 0].round(5)), c='red')
    ax.text(X[0, 0], Y[0, 0], Z[0, 0] + 0.1, str(Z[0, 0].round(5)), c='gray')
    ax.text(X[-1, -1], Y[-1, -1], Z[-1, -1] + 0.1, str(Z[-1, -1].round(5)), c='gray')

    ax.view_init(45, 120)
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                           cmap=cm.RdBu, linewidth=0, antialiased=False)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    fig.colorbar(surf, shrink=0.5, aspect=5)
    # plt.show()
    summary_writer.add_figure(tag, fig, global_step=step)
    return fig

if __name__ == '__main__':
    plot_averaged_runs(args.path)