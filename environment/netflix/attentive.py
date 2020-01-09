import mxnet.ndarray as nd
import numpy as np

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

        with self.name_scope():
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

            self.predictor_net = FeedForward("predictor_", predictor_in_features, config["model"]["predictor"]["layers"])
            self.attention_net = FeedForward("attention_", attention_in_features, config["model"]["attention"]["layers"])

    def _featurize(self, F, x):
        xs_predictor, xs_attention_query, xs_attention_keys, xs_attention_values = [], [], [], []
        feature_ind = 0
        for feature in self.config["features"]:
            x_encoded, feature_ind = self._encode_feature(feature, feature_ind, F, x)

            if "attention-query" in feature["components"]:
                xs_attention_query.append(x_encoded)

            if "attention-key" in feature["components"]:
                xs_attention_keys.append(x_encoded)

            if "attention-value" in feature["components"]:
                xs_attention_values.append(x_encoded)

            if "predictor" in feature["components"]:
                xs_predictor.append(x_encoded)

        x_attention_query = F.concat(*xs_attention_query, dim=2)
        x_attention_keys = F.concat(*xs_attention_keys, dim=2)
        x_attention = F.concat(*[F.tile(x_attention_query, reps=(1, self.sequence_len, 1)), x_attention_keys], dim=2)
        x_attention_values = F.concat(*xs_attention_values, dim=2)
        x_predictor = F.squeeze(F.concat(*xs_predictor, dim=2), axis=1)
        return x_attention, x_attention_values, x_predictor

    def _attention(self, F, x_attention):
        return F.softmax(self.attention_net(x_attention), axis=1)

    def hybrid_forward(self, F, x):
        x_attention, x_attention_values, x_predictor = self._featurize(F, x)
        attention = self._attention(F, x_attention)
        x_attended = F.sum(F.broadcast_mul(x_attention_values, attention), axis=1)
        return self.predictor_net(F.concat(*[x_predictor, x_attended], dim=1))

    def attention(self, x):
        x_attention, x_attention_values, x_predictor = self._featurize(nd, x)
        return self._attention(nd, x_attention).squeeze(axis=2)
