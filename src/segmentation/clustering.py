"""Choosing K and fitting K-Means."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score

RANDOM_STATE = 42


def evaluate_k(X: np.ndarray, k_range=range(2, 11), random_state: int = RANDOM_STATE) -> pd.DataFrame:
    """Fit K-Means for each k and collect inertia (elbow) plus validity scores.

    - inertia: within-cluster sum of squares; look for the "elbow"
    - silhouette: -1..1, higher = tighter, better separated clusters
    - davies_bouldin: lower is better
    - calinski_harabasz: higher is better
    """
    rows = []
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=random_state).fit(X)
        rows.append(
            {
                "k": k,
                "inertia": km.inertia_,
                "silhouette": silhouette_score(X, km.labels_),
                "davies_bouldin": davies_bouldin_score(X, km.labels_),
                "calinski_harabasz": calinski_harabasz_score(X, km.labels_),
            }
        )
    return pd.DataFrame(rows).set_index("k")


def find_elbow(k_values, inertias) -> int:
    """Elbow = point with the largest distance to the straight line joining the
    first and last points of the (normalized) inertia curve (the "kneedle" idea)."""
    k = np.asarray(list(k_values), dtype=float)
    y = np.asarray(list(inertias), dtype=float)
    kn = (k - k.min()) / (k.max() - k.min())
    yn = (y - y.min()) / (y.max() - y.min())
    # line from (0, 1) to (1, 0) for a decreasing curve -> distance ∝ |x + y - 1|
    dist = np.abs(kn + yn - 1)
    return int(k[np.argmax(dist)])


def choose_k(scores: pd.DataFrame, min_k: int = 3, max_k: int = 8, tolerance: float = 0.9) -> tuple[int, str]:
    """Combine the elbow and silhouette methods into one decision.

    Silhouette on RFM data almost always peaks at k=2 ("good" vs "bad"
    customers), which is not actionable for marketing, so we search in
    [min_k, max_k]. Among k values whose silhouette is within `tolerance` of the
    best in that range, pick the one closest to the elbow.
    """
    elbow = find_elbow(scores.index, scores["inertia"])
    window = scores.loc[(scores.index >= min_k) & (scores.index <= max_k), "silhouette"]
    good = window[window >= tolerance * window.max()]
    k = int(min(good.index, key=lambda kk: (abs(kk - elbow), -good[kk])))
    reason = (
        f"elbow at k={elbow}; best silhouette in [{min_k},{max_k}] is "
        f"{window.max():.3f} at k={int(window.idxmax())}; k={k} has silhouette "
        f"{scores.loc[k, 'silhouette']:.3f} and is the closest to the elbow among "
        f"k values within {tolerance:.0%} of the best silhouette."
    )
    return k, reason


def fit_kmeans(X: np.ndarray, k: int, random_state: int = RANDOM_STATE) -> KMeans:
    return KMeans(n_clusters=k, n_init=20, random_state=random_state).fit(X)


def pca_project(X: np.ndarray, n_components: int = 2) -> tuple[np.ndarray, PCA]:
    pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
    return pca.fit_transform(X), pca
