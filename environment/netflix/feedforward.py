import mxnet.gluon as gluon
from environment.netflix.utils import *

DENSE = "dense"
MULTIPLY = "multiply"


class FeedForward(gluon.HybridBlock):

    def __init__(self, prefix, in_features, layers_config, tied_to=None):
        super(FeedForward, self).__init__(prefix=prefix)

        with self.name_scope():
            for i, layer in enumerate(layers_config):
                if layer.get("type", DENSE) == DENSE:
                    out_features = layer["out_features"]
                    self.register_child(gluon.nn.Dense(out_features, in_units=in_features, flatten=False, params=block_params_if_any(self.tied_layer(tied_to, i))), name=self.layer_name(i))
                    if "activation" in layer:
                        self.register_child(gluon.nn.Activation(layer["activation"]))
                    in_features = out_features
                else:
                    raise ValueError(layer)

    def hybrid_forward(self, F, x):
        for layer in self._children.values():
            x = layer(x)
        return x

    def layer_name(self, layer_index):
        return "layer-{}".format(layer_index)

    def tied_layer(self, block, layer_index):
        if block is None:
            return None
        return block._children[self.layer_name(layer_index)]
