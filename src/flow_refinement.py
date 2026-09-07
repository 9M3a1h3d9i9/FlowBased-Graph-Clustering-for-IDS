"""Flow-based local cut refinement using the official LocalGraphClustering API.

This module deliberately does not implement NetworkX minimum-cut fallbacks as
surrogates for MQI or other published flow-improvement algorithms.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
from scipy.sparse import csr_matrix


def _graph_local(A: csr_matrix):
    """Create the LocalGraphClustering GraphLocal object from a sparse graph."""
    from localgraphclustering import GraphLocal

    if not isinstance(A, csr_matrix):
        A = A.tocsr()
    if (A != A.T).nnz != 0:
        raise ValueError("Flow refinement requires a symmetric graph")
    return GraphLocal.from_sparse_adjacency(A)


def mqi_refine(A: csr_matrix, ref_nodes: Iterable[int]) -> tuple[np.ndarray, float]:
    """Run the published Max Flow Quotient-cut Improvement (MQI) algorithm."""
    seeds = np.asarray(sorted(set(int(i) for i in ref_nodes)), dtype=np.int64)
    if seeds.size == 0:
        raise ValueError("ref_nodes must contain at least one node")
    if np.any(seeds < 0) or np.any(seeds >= A.shape[0]):
        raise ValueError("ref_nodes contains an invalid node index")

    from localgraphclustering import MQI

    result = MQI(_graph_local(A), seeds.tolist())
    cluster_nodes = np.asarray(result[0], dtype=np.int64)
    conductance = float(result[1])
    return cluster_nodes, conductance


def simple_local_refine(
    A: csr_matrix,
    ref_nodes: Iterable[int],
    delta: float = 0.3,
    relcondflag: bool = True,
    check_connectivity: bool = True,
) -> tuple[np.ndarray, float]:
    """Run LocalGraphClustering's strongly-local flow-based SimpleLocal method."""
    seeds = np.asarray(sorted(set(int(i) for i in ref_nodes)), dtype=np.int64)
    if seeds.size == 0:
        raise ValueError("ref_nodes must contain at least one node")
    if np.any(seeds < 0) or np.any(seeds >= A.shape[0]):
        raise ValueError("ref_nodes contains an invalid node index")

    from localgraphclustering import SimpleLocal

    result = SimpleLocal(
        _graph_local(A),
        seeds.tolist(),
        delta=delta,
        relcondflag=relcondflag,
        check_connectivity=check_connectivity,
    )
    cluster_nodes = np.asarray(result[0], dtype=np.int64)
    conductance = float(result[1])
    return cluster_nodes, conductance


def refine_from_labels(
    A: csr_matrix,
    labels: np.ndarray,
    method: str = "mqi",
    **kwargs,
) -> tuple[np.ndarray, float]:
    """Refine the cluster represented by label 1 without using ground truth.

    The supplied labels must come from an unsupervised baseline such as
    K-Means or Spectral Clustering. Ground-truth labels are intentionally not
    accepted here, preventing evaluation leakage.
    """
    labels = np.asarray(labels)
    if labels.ndim != 1 or len(labels) != A.shape[0]:
        raise ValueError("labels must be a 1-D array with one entry per graph node")
    ref_nodes = np.flatnonzero(labels == 1)
    if ref_nodes.size == 0:
        raise ValueError("labels contain no nodes assigned to cluster 1")

    if method == "mqi":
        return mqi_refine(A, ref_nodes)
    if method == "simple_local":
        return simple_local_refine(A, ref_nodes, **kwargs)
    raise ValueError("method must be 'mqi' or 'simple_local'")
