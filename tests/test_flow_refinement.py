import numpy as np
import pytest
from scipy.sparse import csr_matrix

from src.flow_refinement import mqi_refine, simple_local_refine


def small_graph():
    # Two dense triangles joined by one weak bridge.
    A = np.array(
        [
            [0, 1, 1, 0, 0, 0],
            [1, 0, 1, 0, 0, 0],
            [1, 1, 0, 1, 0, 0],
            [0, 0, 1, 0, 1, 1],
            [0, 0, 0, 1, 0, 1],
            [0, 0, 0, 1, 1, 0],
        ],
        dtype=float,
    )
    return csr_matrix(A)


def test_mqi_rejects_empty_seeds():
    with pytest.raises(ValueError, match="at least one"):
        mqi_refine(small_graph(), [])


def test_mqi_rejects_invalid_seed():
    with pytest.raises(ValueError, match="invalid"):
        mqi_refine(small_graph(), [99])


def test_mqi_rejects_asymmetric_graph():
    A = small_graph().tolil()
    A[0, 1] = 0
    with pytest.raises(ValueError, match="symmetric"):
        mqi_refine(A.tocsr(), [0, 1, 2])


def test_mqi_runs_when_dependency_is_available():
    pytest.importorskip("localgraphclustering")
    nodes, conductance = mqi_refine(small_graph(), [0, 1, 2])
    assert nodes.ndim == 1
    assert len(nodes) > 0
    assert np.isfinite(conductance)


def test_simple_local_runs_when_dependency_is_available():
    pytest.importorskip("localgraphclustering")
    nodes, conductance = simple_local_refine(small_graph(), [0, 1, 2])
    assert nodes.ndim == 1
    assert len(nodes) > 0
    assert np.isfinite(conductance)
