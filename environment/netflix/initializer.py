from mxnet.initializer import *


@register
class EmbeddingInitializer(Initializer):
    def __init__(self, scale=0.07):
        super(EmbeddingInitializer, self).__init__(scale=scale)
        self.scale = scale

    def _init_weight(self, _, arr):
        random.uniform(-self.scale, self.scale, out=arr)
        arr[0, :] = 0
