import os
import shutil

from recsim.agents import full_slate_q_agent
from recsim.agents.random_agent import RandomAgent
from recsim.simulator import recsim_gym, environment, runner_lib
from stable_baselines.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec

from environment import *
from agent import WolpAgent, StaticAgent


def create_random_agent(sess, environment, eval_mode, summary_writer=None):
    return RandomAgent(environment.action_space, random_seed=SEED)


def create_good_agent(sess, environment, eval_mode, summary_writer=None):
    return StaticAgent(environment.action_space, 6)


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


def create_wolp_agent_with_ratio(k_ratio=0.1, policy_kwargs=None, action_noise=None):

    def create_wolp_agent(sess, environment, eval_mode, summary_writer=None):
        return WolpAgent(environment, action_space=environment.action_space,
                         k_ratio=k_ratio, policy_kwargs=policy_kwargs,
                         action_noise=action_noise)

    return create_wolp_agent


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
    base_dir = cleanup_dir(str(DOC_NUM) + '_experiment_with_different_k_OU_noise/')

    env = recsim_gym.RecSimGymEnv(
        environment.Environment(UserModel(), DocumentSampler(), DOC_NUM, 1, resample_documents=False),
        clicked_reward
    )

    noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(DOC_NUM), sigma=0.1 * np.ones(DOC_NUM))

    agents = [
               ("random", create_random_agent),
               ("good", create_good_agent),
               ("bad", create_bad_agent),
               # ('wolpertinger_1_neighbour', create_wolp_agent_with_ratio(0, action_noise=OrnsteinUhlenbeckActionNoise)),
               ('wolpertinger_0.01', create_wolp_agent_with_ratio(0.01, action_noise=noise)),
               ('wolpertinger_0.05', create_wolp_agent_with_ratio(0.05, action_noise=noise)),
               ('wolpertinger_0.1', create_wolp_agent_with_ratio(0.1, action_noise=noise)),
               ('wolpertinger_1.0', create_wolp_agent_with_ratio(1, action_noise=noise)),
    ]
    for agent_name, create_agent_fun in agents:
        env.reset()

        runner = runner_lib.TrainRunner(
            base_dir=base_dir + agent_name,
            create_agent_fn=create_agent_fun,
            env=env,
            max_training_steps=100,
            num_iterations=100
        )
        runner.run_experiment()


if __name__ == "__main__":
    main()
