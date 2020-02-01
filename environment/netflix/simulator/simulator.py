import logging
import os
import shutil

from recsim.agents.random_agent import RandomAgent
from recsim.simulator import recsim_gym, runner_lib

from environment.netflix.simulator.document import MovieSampler
from environment.netflix.simulator.env import NetflixEnvironment, ratings_reward, Clock
from environment.netflix.simulator.user import SessionProvider, UserSampler, UserModel, UserChoiceModel

from environment.netflix.preprocess import Rating


def create_random_agent(sess, environment, eval_mode, summary_writer=None):
    return RandomAgent(environment.action_space, random_seed=42)


def setup_logging():
    logging.basicConfig(format='[%(asctime)s] %(levelname)s %(message)s', level=logging.INFO, datefmt='%I:%M:%S')
    logging.getLogger().setLevel(logging.INFO)


def cleanup_dir(dir_path):
    if os.path.exists(dir_path) and os.path.isdir(dir_path):
        print("Cleaning up {}".format(dir_path))
        shutil.rmtree(dir_path)
    os.makedirs(dir_path)
    return dir_path


def run_experiment():
    slate_size = 3
    clock = Clock()

    movie_sampler = MovieSampler()

    session_provider = SessionProvider(clock)
    user_sampler = UserSampler(session_provider)
    choice_model = UserChoiceModel()
    user_model = UserModel(user_sampler, slate_size, choice_model, session_provider)

    env = recsim_gym.RecSimGymEnv(
        NetflixEnvironment(
            user_model,
            movie_sampler,
            len(movie_sampler.get_available_movies()),
            slate_size,
            resample_documents=False
        ),
        ratings_reward
    )

    agents = [
        ("random", create_random_agent)
    ]

    for agent_name, create_agent_fun in agents:
        base_dir = "experiments/netflix/" + agent_name
        cleanup_dir(base_dir)

        env.reset()

        runner = runner_lib.TrainRunner(
            base_dir=base_dir,
            create_agent_fn=create_agent_fun,
            env=env,
            max_training_steps=100,
            num_iterations=100
        )
        runner.run_experiment()


def main():
    setup_logging()

    run_experiment()


if __name__ == "__main__":
    main()