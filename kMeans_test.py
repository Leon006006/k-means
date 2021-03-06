import kMeans as kMeans

from sklearn.datasets import make_classification
from matplotlib import pyplot as plt
from numpy import where

X, y = make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0,
                           n_clusters_per_class=1, random_state=4)

X = X.T
# plt.plot(X[0, 0], X[1, 0], 'r+')
# plt.plot(X[0, 1], X[1, 1], 'b+')
'''
# Plot for base-data with "correct" classes
# Plot loop taken from
# https://machinelearningmastery.com/clustering-algorithms-with-python/
for class_value in range(2):
    # get column indices for samples with this class
    column_ix = where(y == class_value)
    # create scatter of these samples
    plt.scatter(X[0, column_ix], X[1, column_ix])
plt.show()
'''
# Initialize Class with data-set
test_means = kMeans.kMeans(X)
# Start clustering with number of clusters and iteration steps
found_means = test_means.cluster(4, 15)
