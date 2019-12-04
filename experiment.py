import os
import shutil

from recsim.agents.random_agent import RandomAgent
from recsim.agents import full_slate_q_agent
from recsim.simulator import recsim_gym, environment, runner_lib

from environment import *
from agent import *

def create_random_agent(sess, environment, eval_mode, summary_writer=None):
    return RandomAgent(environment.action_space, random_seed=SEED)


def create_good_agent(sess, environment, eval_mode, summary_writer=None):
    return StaticAgent(environment.action_space, 1)


def create_bad_agent(sess, environment, eval_mode, summary_writer=None):
    return StaticAgent(environment.action_space, 0)

def create_dqn_agent(sess, environment, eval_mode, summary_writer=None):
    kwargs = {
        'observation_space': environment.observation_space,
        'action_space': environment.action_space,
        'summary_writer': summary_writer,
        'eval_mode': eval_mode,
    }
    return full_slate_q_agent.FullSlateQAgent(sess, **kwargs)

def create_wolp_agent(sess, environment, eval_mode, summary_writer=None):
    kwargs = {
        'observation_space': environment.observation_space,
        'action_space': environment.action_space,
        'summary_writer': summary_writer,
        'eval_mode': eval_mode,
    }
    return WolpAgent(sess, **kwargs)

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
    base_dir = cleanup_dir('/tmp/recsim/')

    env = recsim_gym.RecSimGymEnv(
        environment.Environment(UserModel(), DocumentSampler(), DOC_NUM, 1, resample_documents=False),
        clicked_reward
    )

    agents = [('wolpertinger', create_wolp_agent),
              # ("random", create_random_agent),
              # ("good", create_good_agent),
              # ("bad", create_bad_agent),
              # ("DQN", create_dqn_agent)
             ]
    for agent_name, create_agent_fun in agents:
        env.reset()

        runner = runner_lib.TrainRunner(
            base_dir=base_dir + agent_name,
            create_agent_fn=create_agent_fun,
            env=env,
            max_training_steps=5000,
            num_iterations=100
        )
        runner.run_experiment()


if __name__ == "__main__":
    main()
