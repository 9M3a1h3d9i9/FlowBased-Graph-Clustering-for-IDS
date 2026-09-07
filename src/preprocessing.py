"""NSL-KDD preprocessing utilities."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

NSL_KDD_COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins",
    "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label", "difficulty_level",
]

CATEGORICAL_COLUMNS = ["protocol_type", "service", "flag"]


def load_nsl_kdd(path: str) -> pd.DataFrame:
    """Load an NSL-KDD text file using the canonical 43-column schema."""
    df = pd.read_csv(path, header=None)
    if df.shape[1] != len(NSL_KDD_COLUMNS):
        raise ValueError(f"Expected {len(NSL_KDD_COLUMNS)} columns, found {df.shape[1]}")
    df.columns = NSL_KDD_COLUMNS
    return df


def prepare_features(
    df: pd.DataFrame,
    sample_size: int | None = None,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, list[str]]:
    """Encode categorical variables, scale features and optionally stratify-sample rows."""
    work = df.copy()
    y = (work.pop("label").astype(str).str.lower() != "normal").astype(np.int8).to_numpy()
    work = work.drop(columns=["difficulty_level"], errors="ignore")
    X = pd.get_dummies(work, columns=CATEGORICAL_COLUMNS, dtype=float)

    if sample_size is not None and sample_size < len(X):
        from sklearn.model_selection import train_test_split
        X, _, y, _ = train_test_split(
            X, y, train_size=sample_size, stratify=y, random_state=random_state
        )

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y, list(X.columns)


def pca_transform(X: np.ndarray, n_components: int = 50, random_state: int = 42):
    """Fit PCA and return transformed data plus the fitted transformer."""
    if n_components > min(X.shape):
        raise ValueError("n_components cannot exceed min(n_samples, n_features)")
    pca = PCA(n_components=n_components, random_state=random_state)
    return pca.fit_transform(X), pca
