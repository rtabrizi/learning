'''
Goal: cluster a dataset into k different clusters

Dataset is unlabeled --> unsupervised learning approach

each sample is assigned to the cluster with the nearest mean

ITERATIVE OPTIMIZATION:
1. initialize cluster centers (e.g. randomly)
2. Repeat until convergence
    * Update



'''
import numpy as np

np.random.seed(42)

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1-x2)**2))

class KMeans:
    def __init__(self, K=5, max_iters = 100, plot_steps=False):
        self.K = K
        self.max_iters = max_iters
        self.plot_steps = plot_steps

        # list of sample indices for each cluster
        self.clusters = [[] for cluster in range(self.K)]


        # mean feature vector for each cluster
        self.centroids = []

    def predict(self, X):
        self.X = X
        self.n_samples, self.n_features = X.shape

        # initialize centroids
        # array of size self.K
        # for each entry, it picks a random choice between 0 and n_samples
        random_sample_idxs = np.random.choice(self.n_samples, self.K, replace=False)
        self.centroids = [self.X[idx] for idx in random_sample_idxs]

        # optimzation
        for _ in range(self.max_iters):
            # update clusters
            self.clusters = self._create_clusters(self.centroids)
            # update centroids

            # check if converged


        # return cluster labels 
