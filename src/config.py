"""Central configuration for reproducible experiments."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ExperimentConfig:
    random_state: int = 42
    sample_size: int = 15000
    n_pca_components: int = 50
    n_neighbors: int = 15
    n_clusters: int = 2
    symmetrization: str = "union"
    mutual: bool = False
    weighting: str = "binary"
    rbf_sigma: float | None = None

    def validate(self) -> None:
        if self.sample_size <= 0:
            raise ValueError("sample_size must be positive")
        if self.n_pca_components <= 0:
            raise ValueError("n_pca_components must be positive")
        if self.n_neighbors <= 0:
            raise ValueError("n_neighbors must be positive")
        if self.n_clusters < 2:
            raise ValueError("n_clusters must be at least 2")
        if self.symmetrization not in {"union", "mean"}:
            raise ValueError("symmetrization must be 'union' or 'mean'")
        if self.weighting not in {"binary", "rbf"}:
            raise ValueError("weighting must be 'binary' or 'rbf'")
        if self.rbf_sigma is not None and self.rbf_sigma <= 0:
            raise ValueError("rbf_sigma must be positive when provided")
