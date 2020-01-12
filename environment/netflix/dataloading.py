import logging

import torch
import torch.utils.data as td
import numpy as np


REMOVE_TARGET_NONE = "none"
REMOVE_TARGET_ONLY = "target-only"
REMOVE_TARGET_AND_AFTER = "target-and-after"


class NetflixDataLoader(td.dataloader.DataLoader):

    def __init__(self, dataset, config, batch_size, shuffle=False, max_recommended_batch_size=1024):
        if batch_size > max_recommended_batch_size:
            logging.warning(
                "Batch size is too large {}. Lost of copying happening here, so you should apply this to batches of size <= {}".format(batch_size, max_recommended_batch_size))

        self.features_config = config["features"]
        self.target_config = config["target"]
        self.dataset = dataset

        self.ordered_columns = [feature["name"] for feature in self.features_config if feature.get("precompute", False)]
        self.column_indexes = {column: j for j, column in enumerate(self.ordered_columns)}

        sampler = td.sampler.RandomSampler(dataset) if shuffle else td.sampler.SequentialSampler(dataset)
        self.batch_sampler = td.sampler.BatchSampler(sampler, batch_size, False)

        self._target_index = np.vectorize(_random_target_index, otypes=[int])
        self._target_feature = np.vectorize(_target_feature, otypes=[object])
        self._remove_target_only = np.vectorize(_remove_target_only, otypes=[object])
        self._remove_target_and_after = np.vectorize(_remove_target_and_after, otypes=[object])
        self._pad_sequence = np.vectorize(_pad_sequence, otypes=[object])

    def __iter__(self):
        def iterator():
            data = self.dataset.data[self.ordered_columns].values
            for batch_ind in self.batch_sampler:
                yield self.batchify(np.copy(data[batch_ind]))
        return iterator()

    def __len__(self):
        return len(self.batch_sampler)

    def batchify(self, data: np.array):
        ti = self._target_index(data[:, self.column_indexes[self.target_config["source"]]])
        features = []
        for j, feature in enumerate(self.features_config):
            if feature.get("source", ""):
                feature_data = self._target_feature(data[:, self.column_indexes[feature["source"]]], ti)
            else:
                feature_data = data[:, self.column_indexes[feature["name"]]]

            if feature.get("remove_target", REMOVE_TARGET_NONE) == REMOVE_TARGET_ONLY:
                feature_data = self._remove_target_only(feature_data, ti)
            if feature.get("remove_target", REMOVE_TARGET_NONE) == REMOVE_TARGET_AND_AFTER:
                feature_data = self._remove_target_and_after(feature_data, ti)

            padding = feature.get("padding", 0)
            if padding:
                feature_data = self._pad_sequence(feature_data, padding, 1)

            features.append(np.array(feature_data.tolist()).reshape((len(data), -1)))

        target_data = self._target_feature(data[:, self.column_indexes[self.target_config["source"]]], ti).astype(float).reshape((len(data), -1))

        return torch.from_numpy(np.concatenate(features, axis=1)), torch.from_numpy(target_data)


def _random_target_index(values):
    return np.random.randint(len(values))


def _target_feature(values, target_index):
    return values[target_index]


def _remove_target_only(values, target_index):
    if target_index >= 0:
        return values[:int(target_index)] + values[int(target_index) + 1:]
    else:
        return values


def _remove_target_and_after(values, target_index):
    if target_index >= 0:
        return values[:int(target_index)]
    else:
        return values


def _pad_sequence(sequence, padding, dim):
    pad_with = [0] * dim if dim > 1 else 0
    return [pad_with] * (padding - len(sequence)) + sequence[-padding:]
