import numpy as np

class MyKMeans:
    def __init__(self, n_clusters, max_iter=100, tol=0.0001):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol

    def _assign_points(self, X):
        distances = np.linalg.norm(
            X[:, np.newaxis, :] - self.cluster_centers_[np.newaxis, :, :], axis=2
        ) # [samples, clusters]
        self.labels_ = np.argmin(distances, axis=1) # [samples]


    def _update_centroids(self, X):
        updated_centers = np.zeros_like(self.cluster_centers_)

        for i in range(self.n_clusters):
            mask = self.labels_ == i
            if np.any(mask):
                updated_centers[i] = np.mean(X[mask], axis=0)
            else:
                # Reinitialize empty cluster
                updated_centers[i] = X[np.random.randint(0, X.shape[0])]

        return updated_centers

    def fit(self, X: np.ndarray,):
        cluster_centers_indices = np.random.choice(X.shape[0], self.n_clusters) # [n_clusters]
        self.cluster_centers_ = X[cluster_centers_indices, :] # [n_clusters, features]
        for i in range(self.max_iter):
            self._assign_points(X)
            temp = self._update_centroids(X)
            if (np.linalg.norm(temp - self.cluster_centers_) < self.tol):
                break

            self.cluster_centers_ = temp

        return self
    
    def predict(self, X):
        distances = np.linalg.norm(
            X[:, np.newaxis, :] - self.cluster_centers_[np.newaxis, :, :],
            axis=2
        )
        return np.argmin(distances, axis=1)
    
if __name__ == "__main__":
    from sklearn.datasets import load_iris
    from sklearn.cluster import KMeans

    # Load the Iris dataset
    iris = load_iris()

    # Access the features and target variable
    X = iris.data # Features (sepal length, sepal width, petal length, petal width)
    y = iris.target # Target variable (species: 0 for setosa, 1 for versicolor, 2 for virginica)

    kmeans = KMeans(n_clusters=3)

    kmeans.fit(X)

    my_kmeans = MyKMeans(n_clusters=3)
    my_kmeans.fit(X)

    print(kmeans.cluster_centers_)
    print(my_kmeans.cluster_centers_)
