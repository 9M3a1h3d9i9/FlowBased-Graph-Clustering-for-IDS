"""Similarity-graph construction with explicit topology and weighting semantics."""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import kneighbors_graph


def build_knn_graph(
    X: np.ndarray,
    n_neighbors: int = 15,
    mode: str = "connectivity",
    symmetrization: str = "union",
    mutual: bool = False,
    weighting: str = "binary",
    sigma: float | None = None,
) -> csr_matrix:
    """Build a symmetric kNN affinity graph.

    Nodes represent network-flow records; edges represent feature-space proximity.

    Parameters
    ----------
    X:
        Feature representation used to compute nearest neighbors.
    n_neighbors:
        Number of directed nearest neighbors before symmetrization.
    mode:
        ``connectivity`` for binary edges or ``distance`` for Euclidean distances.
        ``distance`` is required internally for RBF weighting.
    symmetrization:
        ``union`` uses element-wise maximum; ``mean`` averages reciprocal edges.
        Ignored when ``mutual=True``.
    mutual:
        If true, keep only reciprocal kNN edges (mutual kNN graph).
    weighting:
        ``binary`` keeps connectivity weights; ``rbf`` converts distances to
        Gaussian affinities in (0, 1].
    sigma:
        RBF bandwidth. If omitted, the median positive directed kNN distance is
        used, making the weighting data-adaptive and reproducible.
    """
    if n_neighbors <= 0:
        raise ValueError("n_neighbors must be positive")
    if n_neighbors >= len(X):
        raise ValueError("n_neighbors must be smaller than the number of samples")
    if symmetrization not in {"union", "mean"}:
        raise ValueError("symmetrization must be 'union' or 'mean'")
    if weighting not in {"binary", "rbf"}:
        raise ValueError("weighting must be 'binary' or 'rbf'")
    if mode not in {"connectivity", "distance"}:
        raise ValueError("mode must be 'connectivity' or 'distance'")

    graph_mode = "distance" if weighting == "rbf" else mode
    directed = kneighbors_graph(
        X,
        n_neighbors=n_neighbors,
        mode=graph_mode,
        include_self=False,
        n_jobs=-1,
    ).tocsr()

    if weighting == "rbf":
        positive_distances = directed.data[directed.data > 0]
        if sigma is None:
            sigma = float(np.median(positive_distances)) if len(positive_distances) else 1.0
        if sigma <= 0:
            raise ValueError("sigma must be positive")
        directed.data = np.exp(-(directed.data ** 2) / (2.0 * sigma ** 2))

    if mutual:
        # A reciprocal edge exists only when both directed kNN relations exist.
        A = directed.minimum(directed.T)
    elif symmetrization == "union":
        A = directed.maximum(directed.T)
    else:
        A = (directed + directed.T) * 0.5

    A.eliminate_zeros()
    return A.tocsr()


def graph_statistics(A: csr_matrix) -> dict:
    """Return basic diagnostics used to validate graph construction."""
    import scipy.sparse.csgraph as csgraph

    n_components, _ = csgraph.connected_components(A, directed=False)
    degrees = np.asarray(A.getnnz(axis=1)).ravel()
    return {
        "nodes": int(A.shape[0]),
        "edges": int(A.nnz // 2),
        "connected_components": int(n_components),
        "min_degree": int(degrees.min()) if len(degrees) else 0,
        "max_degree": int(degrees.max()) if len(degrees) else 0,
        "mean_degree": float(degrees.mean()) if len(degrees) else 0.0,
    }
