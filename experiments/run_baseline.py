"""End-to-end reproducible baseline experiment for the IDS graph pipeline.

Usage
-----
python -m experiments.run_baseline --data Data/KDDTrain+.txt

The script never uses ground-truth labels to construct clustering seeds. True
labels are consumed only by the evaluation layer after predictions are made.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.clustering import kmeans, spectral
from src.config import ExperimentConfig
from src.evaluation import binary_metrics, silhouette
from src.flow_refinement import refine_from_labels
from src.graph import build_knn_graph, graph_statistics
from src.preprocessing import load_nsl_kdd, pca_transform, prepare_features


def _evaluate(name: str, y_true: np.ndarray, labels: np.ndarray, X: np.ndarray) -> dict:
    metrics = binary_metrics(y_true, labels)
    metrics["silhouette"] = silhouette(X, labels)
    metrics["method"] = name
    return metrics


def run(data_path: str, output_path: str, config: ExperimentConfig) -> dict:
    df = load_nsl_kdd(data_path)
    X_scaled, y, _ = prepare_features(
        df,
        sample_size=config.sample_size,
        random_state=config.random_state,
    )
    X_pca, pca = pca_transform(
        X_scaled,
        n_components=config.n_pca_components,
        random_state=config.random_state,
    )
    A = build_knn_graph(
        X_pca,
        n_neighbors=config.n_neighbors,
        symmetrization=config.symmetrization,
    )

    km_labels = kmeans(X_pca, config.n_clusters, config.random_state)
    sp_labels = spectral(A, config.n_clusters, config.random_state)

    results = [
        _evaluate("kmeans", y, km_labels, X_pca),
        _evaluate("spectral", y, sp_labels, X_pca),
    ]

    # MQI is a local refinement algorithm, not a generic two-way partitioner.
    # We therefore refine the cluster labelled 1 by the unsupervised baseline
    # and evaluate that returned local set as a binary prediction.
    for baseline_name, labels in (("kmeans", km_labels), ("spectral", sp_labels)):
        cluster_nodes, conductance = refine_from_labels(A, labels, method="mqi")
        refined = np.zeros(A.shape[0], dtype=int)
        refined[cluster_nodes] = 1
        row = _evaluate(f"mqi_from_{baseline_name}", y, refined, X_pca)
        row["conductance"] = conductance
        results.append(row)

    payload = {
        "config": {
            "random_state": config.random_state,
            "sample_size": config.sample_size,
            "n_pca_components": config.n_pca_components,
            "n_neighbors": config.n_neighbors,
            "n_clusters": config.n_clusters,
            "symmetrization": config.symmetrization,
        },
        "input": {"path": str(data_path), "rows": int(len(df))},
        "preprocessing": {
            "scaled_features": int(X_scaled.shape[1]),
            "pca_features": int(X_pca.shape[1]),
            "pca_explained_variance_ratio_sum": float(pca.explained_variance_ratio_.sum()),
        },
        "graph": graph_statistics(A),
        "results": results,
    }

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="NSL-KDD text file")
    parser.add_argument("--output", default="results/baseline_mqi.json")
    parser.add_argument("--sample-size", type=int, default=15000)
    parser.add_argument("--pca", type=int, default=50)
    parser.add_argument("--k", type=int, default=15)
    args = parser.parse_args()

    config = ExperimentConfig(
        sample_size=args.sample_size,
        n_pca_components=args.pca,
        n_neighbors=args.k,
    )
    payload = run(args.data, args.output, config)
    for row in payload["results"]:
        print(row)


if __name__ == "__main__":
    main()
