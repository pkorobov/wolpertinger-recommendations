import unittest

import torch
import numpy as np

from environment.netflix.attentive import AttentiveRecommender


class TestRecommender(unittest.TestCase):

    def setUp(self):
        self.recommender = AttentiveRecommender({
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

        for name, param in self.recommender.named_parameters():
            if name == "feature_modules.feature_2.weight":
                param.data = torch.from_numpy(np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]))
            elif name == "predictor_net.linear-0.weight":
                param.data = torch.from_numpy(np.array([[0.0, 1.0, 0.0, 1.0, 0.0]]))
            elif name == "predictor_net.linear-0.bias":
                param.data = torch.from_numpy(np.array([1.0]))
            elif name == "attention_net.linear-0.weight":
                param.data = torch.from_numpy(np.array([[1.0, 0.0, 1.0, 0.0, 1.0]]))
            elif name == "attention_net.linear-0.bias":
                param.data = torch.from_numpy(np.array([2.0]))
            else:
                ValueError()

        self.input = torch.from_numpy(np.array([
            [1.0, 3.0, 0.0, 3.0, 2.0],
            [0.0, 2.0, 1.0, 3.0, 1.0]
        ]))

    def test_featurize(self):
        x_attention, x_attention_values, x_predictor = self.recommender._featurize(self.input)

        self.assertTrue(np.allclose(x_attention.detach().numpy(), np.array([
            [[1.0, 1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 0.0, 1.0]],
            [[0.0, 0.0, 1.0, 1.0, 0.0], [0.0, 0.0, 1.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0, 0.0]]
        ])))
        self.assertTrue(np.allclose(x_attention_values.detach().numpy(), np.array([
            [[0.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
            [[1.0, 0.0], [1.0, 1.0], [1.0, 0.0]]
        ])))
        self.assertTrue(np.allclose(x_predictor.detach().numpy(), np.array([
            [1.0, 1.0, 1.0],
            [-0.0, 0.0, 1.0]
        ])))

    def test_attention(self):
        attention = self.recommender.attention(self.input)
        self.assertTrue(np.allclose(attention.detach().numpy(), np.array([[0.1553624, 0.4223188, 0.4223188], [0.21194156, 0.57611688, 0.21194156]]), atol=1e-4, rtol=1e-4))

    def test_prediction(self):
        prediction = self.recommender(self.input)
        self.assertTrue(np.allclose(prediction.detach().numpy(), np.array([[1.0 + 1.0 + 0.4223188], [1.0 + 1.0]]), atol=1e-4, rtol=1e-4))
