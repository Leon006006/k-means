import numpy as np

from numpy import random
from numpy import linalg
from numpy import where
from matplotlib import pyplot as plt


class kMeans:
    def __init__(self, data):
        self.data = data
        self.tags = np.zeros((data.shape[1], 1))

    def cluster(self, numb_cluster, iterations):
        """
        :param numb_cluster: number of clusters to find
        :param iterations: maximum number of iterations
        :return: found cluster means
        """
        # Initialize first random cluster points
        means = random.uniform(-1, 1, (2, numb_cluster, iterations))

        for i in range(1, iterations):
            # Measure distance between every data point and
            # Cluster means
            distance = np.zeros((means.shape[1], 1))
            for point in range(self.data.shape[1]):
                for mean in range(means.shape[1]):
                    distance[mean, 0] = linalg.norm(np.subtract(self.data[:, point], means[:, mean, i - 1]))
                # Assign point to closest cluster
                self.tags[point, 0] = np.argmin(distance[:, 0])

            # Loop to fit the means to assigned points
            for class_value in range(numb_cluster):
                # get column indices for points with this class-tag
                column_ix = where(self.tags == class_value)
                # create scatter of these samples
                plt.scatter(self.data[0, column_ix], self.data[1, column_ix])
                # Sum of vectors from one cluster
                index_array = np.asarray(column_ix[0])
                sum_cluster = np.zeros((2,))
                for point in range(index_array.shape[0]):
                    sum_cluster += self.data[:, index_array[point, ]]
                magnitude_cluster = index_array.shape[0]
                # Compute new means of cluster
                # If cluster is empty keep mean
                if magnitude_cluster == 0:
                    continue
                means[:, class_value, i] = (1 / magnitude_cluster) * sum_cluster
                # Plot new means
                plt.plot(means[0, class_value, i], means[1, class_value, i], 'r*', markersize=20)

            plt.show()

            if linalg.norm(np.subtract(means[:, :, i-1], means[:, :, i])) < 10 ** -5:
                print("Cluster-Means aren't changing anymore")
                return means[:, :, i]

        print("Maximum iteration number reached")
        return means[:, :, iterations-1]
