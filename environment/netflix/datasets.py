import logging

import torch.utils.data as td
from environment.netflix.utils import months_between


class NetflixDataset(td.Dataset):

    def __init__(self, data, config, **kwargs):
        self.data = self._precompute_features(data, config, **kwargs)

    def _precompute_features(self, data, config, **kwargs):
        precomputed = data[[]]
        for feature in config["features"]:
            if not feature.get("precompute", False):
                continue

            feature_name = feature["name"]
            logging.info("Precompute {}".format(feature_name))
            feature_fun = globals()["feature_{}".format(feature_name)]
            precomputed[feature_name] = data["ratings"].map(lambda r: feature_fun(r, feature, **kwargs))
        return precomputed

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data.iloc[i]


def feature_movies(ratings, feature, **kwargs):
    movie_indexes = kwargs["movie_indexes"]
    oov_index = feature["num_embeddings"] - 1
    return [movie_indexes.get(r.movie_id, oov_index - 1) for r in ratings]


def feature_months(ratings, feature, **kwargs):
    start_date = kwargs["start_date"]
    return [(1.0 + months_between(start_date, r.date)) for r in ratings]


def feature_ratings(ratings, feature, **kwargs):
    min_rating = 1
    max_rating = 5
    return [2.0 * (r.rating - (min_rating + max_rating) / 2) / (max_rating - min_rating) for r in ratings]