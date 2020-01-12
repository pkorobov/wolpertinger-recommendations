import numpy as np
import torch
import torch.nn.functional as F

from environment.netflix.feedforward import FeedForward
from environment.netflix.recommender import Recommender


class AttentiveRecommender(Recommender):

    def __init__(self, config):
        super(AttentiveRecommender, self).__init__(config)

        paddings = [feature["padding"] for feature in config["features"] if "padding" in feature]
        assert len(np.unique(paddings)) == 1
        self.sequence_len = paddings[0]

        predictor_in_features = 0
        attention_in_features = 0

        for feature in config["features"]:
            feature_dim = self._register_feature(feature)

            if "attention-query" in feature["components"]:
                attention_in_features += feature_dim

            if "attention-key" in feature["components"]:
                attention_in_features += feature_dim

            if "attention-value" in feature["components"]:
                predictor_in_features += feature_dim

            if "predictor" in feature["components"]:
                predictor_in_features += feature_dim

        self.predictor_net = FeedForward(predictor_in_features, config["model"]["predictor"]["layers"])
        self.attention_net = FeedForward(attention_in_features, config["model"]["attention"]["layers"])

    def _featurize(self, x):
        xs_predictor, xs_attention_query, xs_attention_keys, xs_attention_values = [], [], [], []
        feature_ind = 0
        for feature in self.config["features"]:
            x_encoded, feature_ind = self._encode_feature(feature, feature_ind, x)

            if "attention-query" in feature["components"]:
                xs_attention_query.append(x_encoded)

            if "attention-key" in feature["components"]:
                xs_attention_keys.append(x_encoded)

            if "attention-value" in feature["components"]:
                xs_attention_values.append(x_encoded)

            if "predictor" in feature["components"]:
                xs_predictor.append(x_encoded)

        x_attention_query = torch.cat(xs_attention_query, dim=2)
        x_attention_keys = torch.cat(xs_attention_keys, dim=2)
        x_attention = torch.cat([x_attention_query.repeat(1, self.sequence_len, 1), x_attention_keys], dim=2)
        x_attention_values = torch.cat(xs_attention_values, dim=2)
        x_predictor = torch.cat(xs_predictor, dim=2).squeeze(1)
        return x_attention, x_attention_values, x_predictor

    def _attention(self, x_attention):
        return F.softmax(self.attention_net(x_attention), 1)

    def forward(self, x):
        x_attention, x_attention_values, x_predictor = self._featurize(x)
        attention = self._attention(x_attention)
        x_attended = (x_attention_values * attention).sum(dim=1)
        return self.predictor_net(torch.cat([x_predictor, x_attended], dim=1))

    def attention(self, x):
        x_attention, x_attention_values, x_predictor = self._featurize(x)
        return self._attention(x_attention).squeeze(2)
