from tensorboard.backend.event_processing import event_accumulator
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib as mpl
import os

mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=['orange', 'green', 'red', 'blue', 'gray'])


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


logdir = "logs"
runs = pd.DataFrame()
fig, ax = plt.subplots(2, len(os.listdir(logdir)), figsize=(24, 8), sharey=True)
plt.xlabel('Iterations')
plt.ylabel('Mean episode reward')
for i, agent in enumerate(os.listdir(logdir)):
    runs_num = len(os.listdir(logdir + '/' + agent))
    for j in range(runs_num):
        cur_run = pd.Series(_load_run(logdir + '/' + agent + "/run_%s/train" % j)['AverageEpisodeRewards/train'][1])
        runs['run_%s' % j] = cur_run
    means = runs.mean(axis=1)
    stds = runs.std(axis=1)
    ax[0, i].plot(means.index, means, label=agent, alpha=0.7)
    ax[0, i].fill_between(means.index, means - stds, means + stds, alpha=0.4)
    ax[0, i].legend()

    runs_rolling = runs.rolling(window=30).mean()
    means = runs_rolling.mean(axis=1)
    stds = runs_rolling.std(axis=1)
    ax[1, i].plot(means.index, means, label=agent, alpha=0.7)
    ax[1, i].fill_between(means.index, means - stds, means + stds, alpha=0.4)
    ax[1, i].legend()

plt.setp(ax, ylim=(-5, 30))
plt.savefig('averaged_agents.png')