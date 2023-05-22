__authors__ = ['1638309']
__group__ = 'DLL.10'

import numpy as np
import utils

def distance(X: np.array, C: np.array) -> np.array:
    print("Shape of X:", X.shape)  # Debug print
    print("Shape of C:", C.shape)  # Debug print
    # Compute the squared difference between X and C and sum along the third axis (axis 2)
    squared_diff = (X[:, None, :] - C)**2
    dist = np.sqrt(squared_diff.sum(axis=2))
    return dist



def get_colors(centroids):
    color_dist = utils.get_color_prob(centroids)
    labels = [utils.colors[np.argmax(color_dist[i])] for i in range(color_dist.shape[0])]

    return labels

class KMeans:
    def __init__(self, X, K=1, options=None):
        self.old_centroids = None
        self.labels = None
        self.num_iter = 0
        self.K = K
        self._init_X(X)
        self._init_options(options)

    def _init_X(self, X: np.array):
        if len(X.shape) == 3:
            self.N = X.shape[0] * X.shape[1]
            self.D = X.shape[2]
            self.X = np.reshape(X, (self.N, self.D))
        elif len(X.shape) == 2:
            self.N, self.D = X.shape
            self.X = X
        else:
            raise ValueError("Input data must be a 2D or 3D array")

    def _init_options(self, options=None):
        if options is None:
            options = {}

        options.setdefault('improvement_threshold', 20)

        if 'km_init' not in options:
            options['km_init'] = 'first'
        if 'verbose' not in options:
            options['verbose'] = False
        if 'tolerance' not in options:
            options['tolerance'] = 0
        if 'max_iter' not in options:
            options['max_iter'] = np.inf
        if 'fitting' not in options:
            options['fitting'] = 'WCD'  # within class distance.

        self.options = options

    def _init_centroids(self):
        self.old_centroids = np.array([])
        km_init = self.options['km_init'].lower()

        if km_init == 'first':
            i, j = 1, 1
            self.old_centroids = np.append(self.old_centroids, self.X[0])
            self.old_centroids = np.resize(self.old_centroids, (j, self.D))

            while i < self.X.shape[0] and j < self.K and j < self.X.shape[0]:
                point = np.resize(self.X[i], (1, 3))
                unique = True

                for old_point in self.old_centroids:
                    if np.array_equiv(old_point, point):
                        unique = False
                        break

                if unique:
                    j += 1
                    self.old_centroids = np.append(self.old_centroids, point)
                    self.old_centroids = np.resize(self.old_centroids, (j, 3))
                i += 1

            self.centroids = self.old_centroids.copy()

        elif km_init == 'random':
            self.centroids = np.random.rand(self.K, self.X.shape[1])
            self.old_centroids = self.centroids.copy()

        elif km_init == 'custom':
            idx = np.sort(np.unique(self.X, return_index=True, axis=0)[1])
            self.old_centroids = self.X[idx[(len(idx) - self.K):]]
            self.centroids = self.old_centroids.copy()


    def get_labels(self):
        # Calculate distances from all points to centroids
        dists = distance(self.X, self.centroids)

        # Find the closest centroid for each point
        self.labels = np.argmin(dists, axis=1)

    def get_centroids(self):
        # Create a copy of the current centroids to avoid memory reference issues
        self.old_centroids = self.centroids.copy()

        # Iterate through all centroids and calculate their new positions
        for i in range(self.K):
            # Calculate the mean of all points assigned to the current centroid
            assigned_points = self.X[np.where(i == self.labels)]
            self.centroids[i] = np.mean(assigned_points, dtype=np.float64, axis=0)

    def converges(self):
        # Compare old and current centroids for convergence
        is_converged = np.array_equiv(self.old_centroids, self.centroids)
        return is_converged

    def fit(self):
        self._init_centroids()

        # Continue until maximum number of iterations is reached or convergence is achieved
        while self.num_iter < self.options['max_iter']:
            self.num_iter += 1

            # Update labels and centroids
            self.get_labels()
            self.get_centroids()

            # Check for convergence
            if self.converges():
                break

    def withinClassDistance(self):
        if self.options['fitting'].lower() == 'wcd':
            # Calculate the minimum distance for each point to its centroid and compute the average
            min_dists = distance(self.X, self.centroids).min(axis=1)
            avg_within_class_dist = np.average(np.power(min_dists, 2))

            return avg_within_class_dist
        else:
            raise Exception("In this first assignment, no other fittings are specified for us to code.")

    def find_bestK(self, max_K):
        self.K = 2
        wcd = 0.0

        # Iterate until max_K is reached
        for self.K in range(2, max_K):
            prev_wcd = wcd

            # Fit the model and calculate within-class distance
            self.fit()
            wcd = self.withinClassDistance()

            # Check for significant improvement after the first iteration
            if self.K > 2:
                improvement = 1 - (wcd / prev_wcd)
                threshold = self.options['improvement_threshold'] / 100.0

                # If no significant improvement, return the previous K
                if improvement < threshold:
                    self.K -= 1
                    break