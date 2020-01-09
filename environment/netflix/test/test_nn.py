import unittest

import mxnet as mx
import mxnet.ndarray as nd
import numpy as np

from environment.netflix.attentive import AttentiveRecommender


class TestRecommender(unittest.TestCase):

    def setUp(self):
        self.ctx = mx.cpu()

        self.recommender = AttentiveRecommender({
            "features": [
                {
                    "name": "feature_0",
                    "components": ["predictor", "attention-query"],
                    "encoding": "numeric"
                },
                {
                    "name": "feature_1",
                    "components": ["predictor", "attention-query"],
                    "encoding": "embedding",
                    "num_embeddings": 3,
                    "embedding_dim": 2,
                    "tied_features": ["feature_2"]
                },
                {
                    "name": "feature_2",
                    "components": ["attention-key", "attention-value"],
                    "encoding": "embedding",
                    "num_embeddings": 3,
                    "embedding_dim": 2,
                    "padding": 3,
                    "tied_features": ["feature_1"]
                }
            ],
            "target": {
                "name": "target",
                "target_item_id_column": "feature_1",
                "feature_item_ids_column": "feature_2"
            },
            "model": {
                "predictor": {
                    "layers": [
                        {"out_features": 1}
                    ]
                },
                "attention": {
                    "layers": [
                        {"out_features": 1}
                    ]
                }
            },
            "training": {
                "num_epochs": 1,
                "batch_size": 8
            }
        })

        self.recommender._children["feature_1"].collect_params().initialize(mx.init.Constant(nd.array([[0, 0], [1, 0], [0, 1]], dtype=float)), ctx=self.ctx)
        self.recommender.collect_params()["neurec_attention_dense0_weight"].initialize(mx.init.Constant(nd.array([[1, 1, -1, 1, -1]], dtype=float)), ctx=self.ctx)
        self.recommender.collect_params()["neurec_attention_dense0_bias"].initialize(mx.init.Constant(nd.array([0], dtype=float)), ctx=self.ctx)
        self.recommender.collect_params()["neurec_predictor_dense0_weight"].initialize(mx.init.Constant(nd.array([[1, 1, -1, 1, -1]], dtype=float)), ctx=self.ctx)
        self.recommender.collect_params()["neurec_predictor_dense0_bias"].initialize(mx.init.Constant(nd.array([0], dtype=float)), ctx=self.ctx)

        self.input = nd.concat(
            nd.array([[1], [-1]]),
            nd.array([[1], [2]]),
            nd.array([[1, 2, 0], [1, 3, 0]]),
            nd.array([[1, -1], [1, 0]]),
            dim=1
        )

    def test_init(self):
        self.assertEqual(len(self.recommender.predictor_net._children), 1)
        self.assertEqual(len(self.recommender.attention_net._children), 1)
        self.assertEqual(self.recommender._children["feature_1"].weight.shape, (3, 2))
        self.assertEqual(self.recommender._children["feature_2"].weight.shape, (3, 2))
        self.assertEqual(self.recommender._children["feature_2"].weight, self.recommender._children["feature_2"].weight)

    def test_featurize(self):
        x_attention, x_attention_values, x_predictor = self.recommender._featurize(nd, self.input)

        self.assertTrue(np.allclose(x_attention.asnumpy(), np.array([
            [[1, 1, 0, 1, 0], [1, 1, 0, 0, 1], [1, 1, 0, 0, 0]],
            [[-1, 0, 1, 1, 0], [-1, 0, 1, 0, 1], [-1, 0, 1, 0, 0]]
        ])))
        self.assertTrue(np.allclose(x_attention_values.asnumpy(), np.array([
            [[1, 0], [0, 1], [0, 0]],
            [[1, 0], [0, 1], [0, 0]]
        ])))
        self.assertTrue(np.allclose(x_predictor.asnumpy(), np.array([
            [1, 1, 0],
            [-1, 0, 1]
        ])))

    def test_attention(self):
        attention = self.recommender.attention(self.input)
        self.assertTrue(np.allclose(attention.asnumpy(), np.array([[0.6652, 0.0900, 0.2447], [0.6652, 0.0900, 0.2447]]), atol=1e-4, rtol=1e-4))

    def test_prediction(self):
        prediction = self.recommender(self.input)
        print(prediction)
        self.assertTrue(np.allclose(prediction.asnumpy(), np.array([[2 + 0.6652 - 0.0900], [-2 + 0.6652 - 0.0900]]), atol=1e-4, rtol=1e-4))
