import shutil

from recsim.agents import full_slate_q_agent
from recsim.agents.random_agent import RandomAgent
from recsim.simulator import recsim_gym, environment
from plots import plot_averaged_runs
from tensorboardX import SummaryWriter
import os
from environment import *
from agent import WolpertingerRecSim, OptimalAgent
from datetime import datetime

RUNS = 5
MAX_TRAINING_STEPS = 15
NUM_ITERATIONS = 3000
EVAL_EPISODES = 100


def create_random_agent(sess, environment, eval_mode, summary_writer=None):
    return RandomAgent(environment.action_space)

def create_optimal_agent(sess, environment, eval_mode, summary_writer=None):
    return OptimalAgent(environment)

def create_wolp_agent_with_ratio(k_ratio=0.1, policy_kwargs=None, action_noise=None, **kwargs):
    def create_wolp_agent(sess, environment, eval_mode, summary_writer=None):
        return WolpertingerRecSim(environment, action_space=environment.action_space,
                                  k_ratio=k_ratio, action_noise=action_noise, summary_writer=summary_writer,
                                  eval_mode=eval_mode, **kwargs)
    return create_wolp_agent


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
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

    agents = [
                # ('Wolpertinger ' + "(" + str(num_actions(DOC_NUM, 0.1)) + "NN, normal noise)",
                #  create_wolp_agent_with_ratio(0.1, **parameters)),
                ('Wolpertinger ' + "(" + str(num_actions(DOC_NUM, 0.05)) + "NN, normal noise)",
                 create_wolp_agent_with_ratio(0.33, **parameters)),
                ("Optimal", create_optimal_agent),
    ]

    for agent, create_function in agents:
        for run in range(RUNS):
            summary_writer = SummaryWriter(base_dir + "/{}/run_{}/train".format(agent, run))
            agent = create_function(None, env, eval_mode=False, summary_writer=summary_writer)
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
