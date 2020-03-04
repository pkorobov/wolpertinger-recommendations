import shutil

from recsim.agents import full_slate_q_agent
from recsim.agents.random_agent import RandomAgent
from recsim.simulator import recsim_gym, environment
from plots import plot_averaged_runs
from tensorboardX import SummaryWriter
import os
from environment import *
from agent import WolpertingerRecSim, StaticAgent
from datetime import datetime

RUNS = 5
MAX_TRAINING_STEPS = 15
NUM_ITERATIONS = 3000
EVAL_EPISODES = 100

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

start_time = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')


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
    num_actions = lambda actions, k_ratio: max(1, int(actions * k_ratio))

    base_dir = 'logs/' + config.ENV_PARAMETERS['kind'] + '/' + start_time + '/'
    os.makedirs(base_dir)

    agent_class = WolpertingerRecSim
    config.init_w()
    parameters = config.parameters

    for run in range(RUNS):
        summary_writer = SummaryWriter(base_dir + "/agent/run_{}/train".format(run))
        agent = agent_class(env, action_space=env.action_space,
                               k_ratio=0.33, summary_writer=summary_writer,
                               eval_mode=False, **parameters)
        step_number = 0
        total_reward = 0.
        observation = env.reset()
        while step_number < 60000:
            action = agent.begin_episode(observation)
            episode_reward = 0
            while True:
                observation, reward, done, info = env.step(action)
                total_reward += reward
                step_number += 1
                episode_reward += reward
                if done:
                    break
                else:
                    action = agent.step(reward, observation)
            agent.end_episode(reward, observation)
            summary_writer.add_scalar('AverageEpisodeRewards', np.mean(episode_reward), step_number)
    plot_averaged_runs(base_dir)

if __name__ == "__main__":
    main()
