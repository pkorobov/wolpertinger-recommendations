import argparse
import json
import logging
import pickle

import torch
import torch.utils.tensorboard as tb

import numpy as np
import pandas as pd
import datetime as dt
import sklearn.model_selection as ms

import environment.netflix.model.dataloading as dl
import environment.netflix.model.datasets as ds
import environment.netflix.model.training as nt

from environment.netflix.preprocess import Rating
from environment.netflix.model.attentive import AttentiveRecommender
from environment.netflix.model.utils import *

TRAIN = "train"
INFER = "inference"

LOWER_PERCENTILE = 2.5
UPPER_PERCENTILE = 97.5

START_DATE = dt.datetime(1999, 11, 1, 0, 0)
END_DATE = dt.datetime(2006, 1, 1, 0, 0)

np.random.seed(1243)


def setup_logging():
    logging.basicConfig(format='[%(asctime)s] %(levelname)s %(message)s', level=logging.INFO, datefmt='%I:%M:%S')
    logging.getLogger().setLevel(logging.INFO)


def run_experiment(experiment_dir, experiment, regime):
    config_dir = os.path.join(experiment_dir, experiment)
    config_path = os.path.join(config_dir, "config.json")
    logging.info("Start experiment: " + experiment)

    with(open(config_path)) as config_file:
        config = json.load(config_file)

    start = dt.datetime.now().strftime("%Y-%m-%dT%H:%M")
    with tb.SummaryWriter(os.path.join(experiment_dir, "tensorboard", start)) as writer:
        if regime == TRAIN:
            model = train(config, config_dir, writer)
            torch.save(model.state_dict(), os.path.join(config_dir, "model.params"))
        elif regime == INFER:
            model = _create_recommender(config)
            model.load_state_dict(torch.load(os.path.join(config_dir, "model.params")))
        else:
            raise ValueError(regime)

    config = evaluate(model, config)

    with open(config_path, "w") as config_file:
        logging.info("Saving evaluation results in config at " + config_path)
        json.dump(config, config_file, indent=2)


def read_movie_indexes(config):
    movie_frequency_path = os.path.join(config["input"]["processed"]["dir"], "movie_frequency.pkl")
    with open(movie_frequency_path, "rb") as movie_frequency_file:
        movie_frequency = pickle.load(movie_frequency_file)
        movie_indexes = {m: j + 1 for j, (m, f) in enumerate(sorted(movie_frequency.items(), key=lambda kv: kv[1], reverse=True))}
        return movie_indexes


def read_train_data(config):
    partitions = config["training"]["partitions"]
    data_parts = list(find_data_parts(config))[:partitions]
    return pd.concat([read_data_part(part) for part in data_parts])


def read_test_data(config):
    partitions = config["evaluation"]["partitions"]
    data_parts = list(find_data_parts(config))[-partitions:]
    return pd.concat([read_data_part(part) for part in data_parts])


def find_data_parts(config):
    processed_input_config = config["input"]["processed"]

    data_dir = processed_input_config["dir"]
    partition_prefix = processed_input_config["prefix"]

    for this_dir, sub_dirs, files in os.walk(data_dir):
        for file_name in files:
            if file_name.startswith(partition_prefix):
                yield os.path.join(this_dir, file_name)


def read_data_part(part_path):
    logging.info("Reading data at " + part_path)
    return pd.read_pickle(part_path)


def train(config, experiment_dir, writer):
    movie_indexes = read_movie_indexes(config)
    data = read_train_data(config)

    dataset_args = dict(movie_indexes=movie_indexes, start_date=START_DATE, end_date=END_DATE)

    train_users, val_users = ms.train_test_split(data.index.unique(), test_size=config["training"]["validation_fraction"], random_state=42)
    train_subset = data.loc[train_users]
    val_subset = data.loc[val_users]

    train_loader = dl.NetflixDataLoader(ds.NetflixDataset(train_subset, config, **dataset_args), config, batch_size=config["training"]["batch_size"], shuffle=False)
    val_loader = dl.NetflixDataLoader(ds.NetflixDataset(val_subset, config, **dataset_args), config, batch_size=config["training"]["batch_size"], shuffle=False)

    logging.info("Training examples: {} Validation examples: {}".format(len(train_loader.dataset), len(val_loader.dataset)))
    model = _create_recommender(config)

    checkpoint_dir = ensure_dir(os.path.join(experiment_dir, "checkpoint"))
    nt.train(model, config, train_loader, val_loader, batches_in_step=10, batches_in_epoch=100, writer=writer, checkpoint_dir=checkpoint_dir)

    example = next(iter(val_loader))[0][[0]]
    writer.add_graph(model, example)

    movie_embeddings = model.feature_modules["movies"].weight.data
    writer.add_embedding(movie_embeddings, _read_movie_meta(config)[:movie_embeddings.shape[0]])

    return model


def _create_recommender(config):
    model = AttentiveRecommender(config)
    nt.initialize(model)
    return model


def _read_movie_meta(config):
    movie_indexes = read_movie_indexes(config)
    movie_meta = []
    with open(os.path.join(config["input"]["raw"]["dir"], "movie_titles.csv"), encoding="ISO-8859-1") as movie_titles_file:
        for line in movie_titles_file:
            movie_id, year, title = line.split(",", 2)
            movie_meta.append((movie_indexes[int(movie_id)], "{} ({})".format(title.strip(), year)))
    return ["PADDING"] + [title for index, title in sorted(movie_meta)]


def evaluate(model, config, metrics_section="metrics"):
    data = read_test_data(config)
    movie_indexes = read_movie_indexes(config)

    dataset_args = dict(movie_indexes=movie_indexes, start_date=START_DATE, end_date=END_DATE)

    n_samples = config["evaluation"]["bootstrap_samples"]
    logging.info("Start bootstrap validation with {} samples using a dataset with {} examples".format(n_samples, len(data)))
    loss = []
    for j in range(n_samples):
        sample = data.sample(frac=1, replace=True)
        eval_loader = dl.NetflixDataLoader(ds.NetflixDataset(sample, config, **dataset_args), config, batch_size=config["training"]["batch_size"], shuffle=False)
        loss.append(nt.validate(model, config, eval_loader))

    config["evaluation"][metrics_section] = [
        {"metric": "loss", "mean": np.mean(loss), "upper": np.percentile(loss, UPPER_PERCENTILE), "lower": np.percentile(loss, LOWER_PERCENTILE)}
    ]

    logging.info("Evaluation results:\n" + pd.DataFrame(config["evaluation"][metrics_section]).set_index("metric").to_string())

    return config


def main():
    setup_logging()

    parser = argparse.ArgumentParser()
    parser.add_argument("--regime", help="Training or inference", choices=[TRAIN, INFER], default=TRAIN)
    parser.add_argument("--experiment-dir", help="Directory where experiment files are located", type=str, required=True)
    parser.add_argument("experiments", help="Experiment name", type=str, nargs="+")
    args = parser.parse_args()

    for experiment in args.experiments:
        run_experiment(args.experiment_dir, experiment, args.regime)


if __name__ == "__main__":
    main()