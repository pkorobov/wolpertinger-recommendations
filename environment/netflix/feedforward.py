import torch.nn as nn

DENSE = "dense"
MULTIPLY = "multiply"


class FeedForward(nn.Module):

    def __init__(self, in_features, layers_config):
        super(FeedForward, self).__init__()

        for i, layer in enumerate(layers_config):
            if layer.get("type", DENSE) == DENSE:
                out_features = layer["out_features"]
                self.add_module("linear-{}".format(i), nn.Linear(in_features, out_features))

                activation = layer.get("activation")
                if activation is None:
                    pass
                elif activation == "tanh":
                    self.add_module("activation-{}".format(i), nn.Tanh())
                else:
                    raise ValueError("Activation not supported: {}".format(activation))

                in_features = out_features
            else:
                raise ValueError("Layer type {} not supported".format(layer.get("type", DENSE)))

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x
