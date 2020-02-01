import torch
import torch.nn as nn
import torch.optim as optim


class Recommender(nn.Module):

    def __init__(self, config):
        super(Recommender, self).__init__()
        self.config = config
        self.feature_modules = nn.ModuleDict()

    def _register_feature(self, feature):
        if feature["encoding"] == "embedding":
            feature_dim = feature["embedding_dim"]

            module_name = self._module_name(feature)
            if module_name not in self.feature_modules:
                self.feature_modules[module_name] = nn.Embedding(feature["num_embeddings"], feature["embedding_dim"])
        elif feature["encoding"] == "numeric":
            feature_dim = 1
        else:
            raise ValueError("Unknown feature encoding in '{}'".format(feature))

        return feature_dim

    def _encode_feature(self, feature, feature_ind, x):
        feature_width = feature.get("padding", 1)

        x_feature = x.narrow(dim=1, start=feature_ind, length=feature_width)
        if feature["encoding"] == "embedding":
            x_feature = x_feature.long()

        module_name = self._module_name(feature)
        if module_name in self.feature_modules:
            x_encoded = self.feature_modules[module_name](x_feature)
        else:
            x_encoded = torch.reshape(x_feature, shape=(x.shape[0], feature_width, 1))

        next_feature_ind = feature_ind + feature_width
        return x_encoded, next_feature_ind

    def _module_name(self, feature):
        return feature["module_name"] if "module_name" in feature else feature["name"]

    def create_optimizer(self):
        return optim.Adam(self.parameters(), lr=self.config["training"]["learning_rate"])


