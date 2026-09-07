"""Similarity-graph construction with explicit symmetry semantics."""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import kneighbors_graph


def build_knn_graph(
    X: np.ndarray,
    n_neighbors: int = 15,
    mode: str = "connectivity",
    symmetrization: str = "union",
) -> csr_matrix:
    """Build an undirected kNN graph.

    Nodes represent network-flow records; edges represent feature-space proximity.
    ``union`` uses A=max(A,A.T), while ``mean`` averages reciprocal edges.
    """
    if n_neighbors >= len(X):
        raise ValueError("n_neighbors must be smaller than the number of samples")
    A = kneighbors_graph(
        X, n_neighbors=n_neighbors, mode=mode, include_self=False, n_jobs=-1
    ).tocsr()
    if symmetrization == "union":
        A = A.maximum(A.T)
    elif symmetrization == "mean":
        A = (A + A.T) * 0.5
    else:
        raise ValueError("symmetrization must be 'union' or 'mean'")
    A.eliminate_zeros()
    return A.tocsr()


def graph_statistics(A: csr_matrix) -> dict:
    """Return basic diagnostics used to validate graph construction."""
    import scipy.sparse.csgraph as csgraph

    n_components, labels = csgraph.connected_components(A, directed=False)
    degrees = np.asarray(A.getnnz(axis=1)).ravel()
    return {
        "nodes": int(A.shape[0]),
        "edges": int(A.nnz // 2),
        "connected_components": int(n_components),
        "min_degree": int(degrees.min()) if len(degrees) else 0,
        "max_degree": int(degrees.max()) if len(degrees) else 0,
        "mean_degree": float(degrees.mean()) if len(degrees) else 0.0,
    }
