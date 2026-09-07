import pandas as pd

from experiments.analyze_graph_sweep import rank_sweep


def test_rank_sweep_prefers_connected_multi_metric_configuration(tmp_path):
    rows = [
        {
            "method": "spectral", "k": 10, "symmetrization": "union",
            "mutual": False, "weighting": "binary", "connected_components": 1,
            "edges": 40, "mean_degree": 8, "f1": 0.90, "ari": 0.80,
            "nmi": 0.85, "balanced_accuracy": 0.88, "silhouette": 0.30,
        },
        {
            "method": "spectral", "k": 5, "symmetrization": "union",
            "mutual": True, "weighting": "rbf", "connected_components": 2,
            "edges": 20, "mean_degree": 4, "f1": 0.99, "ari": 0.99,
            "nmi": 0.99, "balanced_accuracy": 0.99, "silhouette": 0.99,
        },
    ]
    source = tmp_path / "sweep.csv"
    output = tmp_path / "ranked.csv"
    pd.DataFrame(rows).to_csv(source, index=False)

    ranked = rank_sweep(str(source), str(output), top_k=1)
    assert len(ranked) == 1
    assert int(ranked.iloc[0]["k"]) == 10
    assert int(ranked.iloc[0]["connected_components"]) == 1
    assert output.exists()
