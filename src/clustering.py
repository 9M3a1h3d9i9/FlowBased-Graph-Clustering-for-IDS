"""Baseline clustering methods used by the experiment."""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.cluster import KMeans, SpectralClustering


def kmeans(X: np.ndarray, n_clusters: int = 2, random_state: int = 42) -> np.ndarray:
    return KMeans(n_clusters=n_clusters, random_state=random_state, n_init=20).fit_predict(X)


def spectral(A: csr_matrix, n_clusters: int = 2, random_state: int = 42) -> np.ndarray:
    """Run spectral clustering on an explicitly symmetric affinity graph."""
    model = SpectralClustering(
        n_clusters=n_clusters,
        affinity="precomputed",
        random_state=random_state,
        n_init=20,
    )
    return model.fit_predict(A)
