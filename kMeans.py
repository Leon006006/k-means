import numpy as np
import matplotlib

from numpy import random
from numpy import linalg
from numpy import where
from matplotlib import pyplot as plt

class kMeans:
    def __init__(self, data):
        self.data = data
        self.tags = np.zeros((data.shape[1], 1))

    def cluster(self):
        # Initialize first random cluster points
        means = random.uniform(-1, 1, (2, 2))
        plt.plot(means[0, 0], means[1, 0], 'b*', markersize=20)
        plt.plot(means[0, 1], means[1, 1], 'r*', markersize=20)

        distance = np.zeros((2, 1))
        for point in range(self.data.shape[1]):
            for mean in range(means.shape[1]):
                distance[mean, 0] = linalg.norm(np.abs(np.subtract(self.data[:, point], means[:, mean])))
            self.tags[point, 0] = np.argmin(distance[:, 0])

        for class_value in range(2):
            # get row indexes for samples with this class
            row_ix = where(self.tags == class_value)
            # create scatter of these samples
            plt.scatter(self.data[0, row_ix], self.data[1, row_ix])
        plt.show()