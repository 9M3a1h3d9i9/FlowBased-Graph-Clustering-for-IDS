"""Research-oriented analysis and ranking of graph sweep results.

The ranking is deliberately multi-metric. It is intended for configuration
screening and hypothesis generation, not final test-set model selection.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import pandas as pd

METRICS = ["f1", "ari", "nmi", "balanced_accuracy", "silhouette"]
CONFIG_COLUMNS = [
    "method", "k", "symmetrization", "mutual", "weighting",
    "connected_components", "edges", "mean_degree",
]


def rank_sweep(input_path: str, output_path: str, top_k: int = 10) -> pd.DataFrame:
    """Rank sweep rows using average ranks across complementary metrics."""
    df = pd.read_csv(input_path)
    required = set(METRICS + CONFIG_COLUMNS)
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if df.empty:
        raise ValueError("Sweep result file is empty")

    candidate = df.copy()
    connected = candidate["connected_components"] == 1
    if connected.any():
        candidate = candidate.loc[connected].copy()

    for metric in METRICS:
        candidate[f"rank_{metric}"] = candidate[metric].rank(
            method="min", ascending=False, na_option="bottom"
        )

    rank_columns = [f"rank_{metric}" for metric in METRICS]
    candidate["mean_metric_rank"] = candidate[rank_columns].mean(axis=1)
    candidate = candidate.sort_values(
        ["mean_metric_rank", "f1", "ari", "nmi", "silhouette"],
        ascending=[True, False, False, False, False],
    ).reset_index(drop=True)
    candidate["overall_rank"] = range(1, len(candidate) + 1)

    result = candidate.head(top_k)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out, index=False, quoting=csv.QUOTE_MINIMAL)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="results/graph_sweep_ranked.csv")
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()
    if args.top_k <= 0:
        raise ValueError("top-k must be positive")
    result = rank_sweep(args.input, args.output, args.top_k)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
