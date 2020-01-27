from Operations import closest_to_target
import numpy as np
import math
import random


class KMeans:

    def __init__(self, n_of_clusters, max_iter, init='kmeans++'):
        """ KMeans class constructor.
        :param y: Data.
        :param n_of_clusters: Number of clusters.
        :param max_iter: Maximum number of iterations.
        """
        self.n_of_clusters = n_of_clusters
        self.max_iter = max_iter
        self.init = init

    def fit(self, x):
        """ Fits the model to the given data.
        :param x: The data.
        """
        self.data = np.array(x)
        self.sample_size, self.n_of_attributes = x.shape
        self.centroids = np.empty(shape=[self.n_of_clusters, self.n_of_attributes])
        if self.sample_size < self.n_of_clusters:
            raise ValueError("Sample size is smaller than the number of clusters.")

        self.initialise_centroids()

        for i in range(self.max_iter):
            new_centroids = np.zeros([self.n_of_clusters, self.n_of_attributes])
            new_centroids_points_per_cluster = [0] * self.n_of_clusters

            for point in self.data:
                index, _ = closest_to_target(point, self.centroids)
                new_centroids[index] += point
                new_centroids_points_per_cluster[index] += 1

            for c in range(self.n_of_clusters):
                new_centroids[c] /= new_centroids_points_per_cluster[c]

            eq = np.array_equal(new_centroids, self.centroids)
            self.centroids = new_centroids

            if eq:
                break

    def initialise_centroids(self):
        """ Initialises the centroids. """

        if self.init == 'kmeans++':
            self.initialise_centroids_kmeans_pp()
            return
        elif self.init == 'random':
            self.initialise_centroids_random()
        else:
            raise TypeError("Unknown init.")

    def initialise_centroids_kmeans_pp(self):
        """ Initializes the centroids using k-means++'s algorithm. """

        for i in range(self.n_of_clusters):
            if not i:
                self.centroids[i] = (random.choice(self.data))
                continue

            point_weights = [math.pow(closest_to_target(x, self.centroids)[1], 2) for x in self.data]
            self.centroids[i] = (random.choices(population=self.data, weights=point_weights)[0])

    def initialise_centroids_random(self):
        """ Initializes the centroids on random data points. """

        rand_indexes = random.sample(range(self.sample_size), self.n_of_clusters)
        for i in range(self.n_of_clusters):
            self.centroids[i] = self.data[i]
