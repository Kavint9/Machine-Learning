from skimage import io
import numpy as np
import sys
import time

class KMeans:
    def __init__(self, n_clusters=5, max_iters=60):
        self.n_clusters = n_clusters
        self.max_iters = max_iters

        # list of indices for each cluster
        # self.clusters = np.array([list() for _ in range(self.n_clusters)]
        #         list of lists to store index of all elements belonging to a cluster
        self.clusters = [list() for _ in range(self.n_clusters)]

        # centers for each cluster
        #         list of RGB values for each centroid
        self.centroids = []

    def predict(self, X):
        self.X = X
        self.n_samples, self.n_features = X.shape

        # Initialize clusters
        random_sample_indices = np.random.choice(self.n_samples, self.n_clusters, replace=False)

        self.centroids = [self.X[index] for index in random_sample_indices]

        # Optimize clusters
        for i in range(self.max_iters):
            if i % 10 == 0:
                print(f'{i}th iteration')
            # Assign samples to closest centroids
            self.clusters = self._create_clusters(self.centroids)

            centroids_prev = self.centroids
            self.centroids = self._get_centroids(self.clusters)

            # check if clusters have changed
            if self._is_converged(centroids_prev, self.centroids):
                break

        return self._get_cluster_labels(self.clusters)

    def _get_cluster_labels(self, clusters):
        # uninitialized array of the number of samples
        labels = np.empty(self.n_samples)
        for cluster_index, cluster in enumerate(clusters):
            for sample_index in cluster:
                labels[sample_index] = cluster_index

        return labels

    def _create_clusters(self, centroids):
        # Assign the samples to the closest centroids to create clusters
        clusters = [list() for _ in range(self.n_clusters)]

        for index, sample in enumerate(self.X):
            centroid_index = self._closest_centroid(sample, centroids)
            clusters[centroid_index].append(index)
        return clusters

    def _closest_centroid(self, sample, centroids):
        #       since squaring a term is monotonic distance square is used for comparison
        return np.argmin(np.sum((sample - centroids) ** 2, axis=1))

    def _get_centroids(self, clusters):
        centroids = np.zeros((self.n_clusters, self.n_features))
        for cluster_index, cluster in enumerate(clusters):
            cluster_mean = np.mean(self.X[cluster], axis=0)
            centroids[cluster_index] = cluster_mean
        return centroids

    def _is_converged(self, centroids_prev, centroids):
        return np.array_equal(centroids_prev, centroids)

    def euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print('Usage is KMeans_image_compression <ip filepath> <k value> <op filepath>')

    ipfile = sys.argv[1]
    n_clusters = int(sys.argv[2])
    opfile = sys.argv[3]
    print(f'KMeans computing {n_clusters} Cluster ')
    print(f'Max Iteration 60. Approx time taken: 10 min')
    image = io.imread(ipfile)

    rows = image.shape[0]
    cols = image.shape[1]

    image = image.reshape(rows * cols, 3)

    inst = KMeans(n_clusters=n_clusters)
    start = time.time()
    labels = inst.predict(image)
    end = time.time()
    labels = labels.astype(int)
    output = inst.centroids[labels]
    compressed_image = output.reshape(rows, cols, 3)
    compressed_image = np.clip(compressed_image.astype('uint8'), 0, 255)
    print(f'Saving compressed image to {opfile}')
    print(f'Time taken: {(end-start)/60}')
    io.imsave(opfile, compressed_image)