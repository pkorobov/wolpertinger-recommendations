import unittest

import numpy as np
import pandas as pd

import environment.netflix.model.dataloading as dl


np.random.seed(12345)


class TestDataLoading(unittest.TestCase):

    def setUp(self):
        self.config = {
            "features": [
                {
                    "name": "feature_0",
                    "components": ["predictor", "attention-query"],
                    "encoding": "numeric",
                    "precompute": True
                },
                {
                    "name": "feature_1",
                    "source": "feature_2",
                    "components": ["predictor", "attention-query"],
                    "encoding": "embedding",
                    "num_embeddings": 4,
                    "embedding_dim": 2,
                    "module_name": "feature_2"
                },
                {
                    "name": "feature_2",
                    "components": ["attention-key", "attention-value"],
                    "encoding": "embedding",
                    "num_embeddings": 4,
                    "embedding_dim": 2,
                    "padding": 3,
                    "remove_target": "target-and-after",
                    "precompute": True
                }
            ],
            "target": {
                "name": "target",
                "source": "feature_2"
            }
        }

        self.data = pd.DataFrame({
            "feature_0": [1.0, -1.0, 1.0, 0.0],
            "feature_2": [[2], [2, 1], [2, 3, 1], [2, 3, 1, 1, 3]]
        })

        self.dataset = object()

        self.loader = dl.NetflixDataLoader(self, self.config, batch_size=1)

    def test_target_index(self):
        target_index = self.loader._target_index(self.data.values[:, self.loader.column_indexes[self.loader.target_config["source"]]])
        self.assertListEqual(target_index.tolist(), [0, 0, 1, 2])

    def test_target_feature(self):
        target_index = np.array([0, 1, 0, 4])
        feature_data = self.data.values[:, self.loader.column_indexes[self.loader.features_config[1]["source"]]]
        target_feature = self.loader._target_feature(feature_data, target_index)
        self.assertListEqual(target_feature.tolist(), [2, 1, 2, 3])

    def test_remove_target_and_after(self):
        target_index = np.array([0, 1, 0, 4])
        feature_data = self.data.values[:, self.loader.column_indexes[self.loader.features_config[2]["name"]]]
        data = self.loader._remove_target_and_after(feature_data, target_index)

        self.assertTrue(np.array_equal(data, [[], [2], [], [2, 3, 1, 1]]))

    def test_remove_target_only(self):
        target_index = np.array([0, 1, 0, 4])
        feature_data = self.data.values[:, self.loader.column_indexes[self.loader.features_config[2]["name"]]]
        data = self.loader._remove_target_only(feature_data, target_index)

        self.assertTrue(np.array_equal(data, [[], [2], [3, 1], [2, 3, 1, 1]]))

    def test_pad_sequence(self):
        target_index = np.array([0, 1, 0, 4])
        feature_data = self.data.values[:, self.loader.column_indexes[self.loader.features_config[2]["name"]]]
        data = self.loader._remove_target_only(feature_data, target_index)
        data = self.loader._pad_sequence(data, 3, 1)

        self.assertListEqual(data.tolist(), [[0, 0, 0], [0, 0, 2], [0, 3, 1], [3, 1, 1]])

    def test_batchify(self):
        # target index : [0, 0, 1, 1]
        x, y = self.loader.batchify(self.data[self.loader.ordered_columns].values)

        self.assertTrue(np.array_equal(x.numpy(), ([
            [+1.0, 2.0, 0.0, 0.0, 0.0],
            [-1.0, 2.0, 0.0, 0.0, 0.0],
            [+1.0, 3.0, 0.0, 0.0, 2.0],
            [+0.0, 3.0, 0.0, 0.0, 2.0]
        ])))
        self.assertTrue(np.allclose(y.numpy(), np.array([[2.0], [2.0], [3.0], [3.0]])))

    def __len__(self):
        return len(self.data)
