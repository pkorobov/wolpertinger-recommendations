import numpy as np
import pyflann

"""
    This class represents a n-dimensional unit cube with a specific number of points embeded.
    Points are distributed uniformly in the initialization. A search can be made using the
    search_point function that returns the k (given) nearest neighbors of the input point.

"""

class KNNSearch:
    def __init__(self, space, embeddings=None):

        # space is a Box space
        n = space.shape[0]
        if embeddings is None:
            self._space = np.eye(n)  # or embedded points in future
            self._low = np.array([-1] * n)
            self._high = np.array([1] * n)

        self._range = self._high - self._low
        self._dimensions = len(self._low)

        self._flann = pyflann.FLANN()
        self._flann.build_index(self._space, algorithm='kdtree')

    def search_point(self, point, k):
        p_in = self.import_point(point)
        search_res, _ = self._flann.nn_index(p_in, k)
        nearest_neighbours = self._space[search_res]
        p_out = []
        for p in nearest_neighbours:
            p_out.append(self.export_point(p))

        if k == 1:
            p_out = [p_out]
        return np.array(p_out)

    def import_point(self, point):
        return (point - self._low) / self._range

    def export_point(self, point):
        return self._low + point * self._range

    def get_space(self):
        return self._space

    def shape(self):
        return self._space.shape

    def get_number_of_actions(self):
        return self.shape()[0]

def get_low_high(embeddings):
    # тут будет функция возвращающая low и high по эмбеддингам
    pass