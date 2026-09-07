import numpy as np

from src.graph import build_knn_graph, graph_statistics


def test_knn_graph_is_symmetric():
    X = np.random.default_rng(42).normal(size=(30, 4))
    A = build_knn_graph(X, n_neighbors=5)
    assert (A != A.T).nnz == 0
    assert A.shape == (30, 30)


def test_graph_statistics():
    X = np.random.default_rng(42).normal(size=(30, 4))
    stats = graph_statistics(build_knn_graph(X, n_neighbors=5))
    assert stats["nodes"] == 30
    assert stats["edges"] > 0
    assert stats["min_degree"] >= 5
