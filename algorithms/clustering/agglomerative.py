import numpy as np

class AgglomerativeClustering:
    def __init__(self, linkage: str = "single"):
        self.linkage = linkage
        self.children_ = None  # shape (n-1, 4)
        self.distances_ = None  # shape (n-1,)

    def _pairwise_distances(self, X: np.array) -> np.array:
        norms = np.sum(X**2, axis=1)
        dists = norms[:, None] + norms[None, :] - 2 * X @ X.T
        return np.sqrt(np.clip(dists, a_min=0, a_max=None))

    def _lance_williams_update(self, d_ik, d_jk, d_ij,
                               size_i, size_j, size_k):

        if self.linkage == "single":
            return np.minimum(d_ik, d_jk)
        if self.linkage == "complete":
            return np.maximum(d_ik, d_jk)
        if self.linkage == "average":
            return (d_ik * size_i + d_jk * size_j) / (size_i + size_j)
        if self.linkage == "ward":
            return (
                (size_i + size_k) * d_ik
                + (size_j + size_k) * d_jk
                - size_k * d_ij
            ) / (size_i + size_j + size_k)

    def fit(self, X: np.ndarray):
        n = X.shape[0]
        dist = np.full((2*n-1, 2*n-1), np.inf)
        dist[:n, :n] = self._pairwise_distances(X)
        if self.linkage == "ward":
            dist[:n, :n] **= 2
        np.fill_diagonal(dist, np.inf)
        active = np.zeros(2*n-1, dtype=bool)
        active[:n] = True
        size = np.zeros(2*n-1, dtype=int)
        size[:n] = 1
        self.children_ = np.empty((n-1, 4), dtype=int)
        self.distances_ = np.empty(n-1)

        for i in range(n-1):
            # find the closest pair among currently active clusters
            active_idx = np.where(active)[0]
            sub = dist[np.ix_(active_idx, active_idx)]
            flat = np.argmin(sub)
            sub_a, sub_b = np.unravel_index(flat, sub.shape)
            a, b = active_idx[sub_a], active_idx[sub_b]

            new_id = n + i

            # record the merge
            self.children_[i] = a, b, new_id, size[a] + size[b]
            merge_dist = dist[a, b]
            self.distances_[i] = np.sqrt(merge_dist) if self.linkage == "ward" else merge_dist

            active[a] = False
            active[b] = False

            # update the distance matrix
            dist[new_id, active] = self._lance_williams_update(
                dist[a, :][active],
                dist[b, :][active],
                dist[a, b],
                size[a], size[b], size[active]
            )
            dist[:, new_id] = dist[new_id]
            size[new_id] = size[a] + size[b]
            active[new_id] = True

        return self