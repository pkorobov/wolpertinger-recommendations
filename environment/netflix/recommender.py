import mxnet.gluon as gluon
import mxnet as mx
from environment.netflix.utils import *


class Recommender(gluon.HybridBlock):

    def __init__(self, config):
        super(Recommender, self).__init__(prefix="neurec_")
        self.config = config

    def _register_feature(self, feature):
        if feature["encoding"] == "embedding":
            feature_dim = feature["embedding_dim"]

            tied_block = None
            for tied_feature in feature.get("tied_features", []):
                if tied_feature in self._children:
                    tied_block = self._children[tied_feature]

            self.register_child(gluon.nn.Embedding(feature["num_embeddings"], feature["embedding_dim"], params=block_params_if_any(tied_block)), feature["name"])
        elif feature["encoding"] == "numeric":
            feature_dim = 1
        else:
            raise ValueError("Unknown feature encoding in '{}'".format(feature))

        return feature_dim

    def _encode_feature(self, feature, feature_ind, F, x):
        feature_width, feature_dim = feature.get("padding", 1), feature.get("dim", 1)
        next_feature_ind = feature_ind + feature_width * feature_dim
        x_feature = F.slice_axis(x, axis=1, begin=feature_ind, end=next_feature_ind)
        x_encoded = self._children[feature["name"]](x_feature) if feature["name"] in self._children else F.reshape(x_feature, shape=(0, feature_width, feature_dim))
        return x_encoded, next_feature_ind

    def hybrid_forward(self, F, x, *args, **kwargs):
        raise NotImplemented()

    def create_optimizer(self):
        return mx.gluon.Trainer(self.collect_params(), mx.optimizer.Adam(self.config["training"]["learning_rate"]))
