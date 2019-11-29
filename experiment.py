import os
import shutil

from recsim.agents.random_agent import RandomAgent
from recsim.simulator import recsim_gym, environment, runner_lib

from environment import *
from agent import *


def create_random_agent(sess, environment, eval_mode, summary_writer=None):
    return RandomAgent(environment.action_space, random_seed=SEED)


def create_good_agent(sess, environment, eval_mode, summary_writer=None):
    return StaticAgent(environment.action_space, 1)


def create_bad_agent(sess, environment, eval_mode, summary_writer=None):
    return StaticAgent(environment.action_space, 0)


def cleanup_dir(dir_path):
    if os.path.exists(dir_path) and os.path.isdir(dir_path):
        print("Cleaning up {}".format(dir_path))
        shutil.rmtree(dir_path)
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

    for agent_name, create_agent_fun in [("random", create_random_agent), ("good", create_good_agent), ("bad", create_bad_agent)]:
        env.reset()

        runner = runner_lib.TrainRunner(
            base_dir=base_dir + agent_name,
            create_agent_fn=create_agent_fun,
            env=env,
            max_training_steps=100,
            num_iterations=20
        )

        runner.run_experiment()


if __name__ == "__main__":
    main()
