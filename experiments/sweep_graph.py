"""Systematic sweep over graph hyperparameters for the IDS baseline.

The sweep evaluates only unsupervised clustering methods. Ground-truth labels
are used after prediction for offline metrics and never influence graph
construction or clustering seeds.

Example
-------
python -m experiments.sweep_graph \
    --data Data/archive/KDDTrain+.txt \
    --k-values 5 10 15 20 30 \
    --symmetrizations union mean
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from src.clustering import kmeans, spectral
from src.config import ExperimentConfig
from src.evaluation import binary_metrics, silhouette
from src.graph import build_knn_graph, graph_statistics
from src.preprocessing import load_nsl_kdd, pca_transform, prepare_features


def run_sweep(
    data_path: str,
    output_path: str,
    k_values: list[int],
    symmetrizations: list[str],
    sample_size: int = 15000,
    n_pca_components: int = 50,
    random_state: int = 42,
) -> list[dict]:
    df = load_nsl_kdd(data_path)
    X_scaled, y, _ = prepare_features(
        df, sample_size=sample_size, random_state=random_state
    )
    X_pca, _ = pca_transform(
        X_scaled, n_components=n_pca_components, random_state=random_state
    )

    rows: list[dict] = []
    for symmetrization in symmetrizations:
        for k in k_values:
            config = ExperimentConfig(
                random_state=random_state,
                sample_size=sample_size,
                n_pca_components=n_pca_components,
                n_neighbors=k,
                symmetrization=symmetrization,
            )
            A = build_knn_graph(
                X_pca,
                n_neighbors=config.n_neighbors,
                symmetrization=config.symmetrization,
            )
            stats = graph_statistics(A)

            methods = {
                "kmeans": kmeans(X_pca, 2, random_state),
                "spectral": spectral(A, 2, random_state),
            }
            for method, labels in methods.items():
                metrics = binary_metrics(y, labels)
                metrics.update(
                    {
                        "method": method,
                        "k": k,
                        "symmetrization": symmetrization,
                        "nodes": stats["nodes"],
                        "edges": stats["edges"],
                        "connected_components": stats["connected_components"],
                        "min_degree": stats["min_degree"],
                        "max_degree": stats["max_degree"],
                        "mean_degree": stats["mean_degree"],
                        "silhouette": silhouette(X_pca, labels),
                    }
                )
                rows.append(metrics)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", default="results/graph_sweep.csv")
    parser.add_argument("--k-values", nargs="+", type=int, default=[5, 10, 15, 20, 30])
    parser.add_argument("--symmetrizations", nargs="+", choices=["union", "mean"], default=["union", "mean"])
    parser.add_argument("--sample-size", type=int, default=15000)
    parser.add_argument("--pca", type=int, default=50)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    rows = run_sweep(
        data_path=args.data,
        output_path=args.output,
        k_values=args.k_values,
        symmetrizations=args.symmetrizations,
        sample_size=args.sample_size,
        n_pca_components=args.pca,
        random_state=args.random_state,
    )
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
