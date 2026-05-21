# ============================================================
# FILE: validation/latent_diagnostics.py
# ============================================================

import warnings
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, clone
from sklearn.manifold import trustworthiness
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise_distances
from scipy.stats import spearmanr


class LatentSpaceDiagnostics(BaseEstimator):

    FRAMEWORK_VERSION = "18.2"

    def __init__(
        self,
        n_neighbors=10,
        n_bootstraps=10,
        sample_fraction=0.8,
        random_state=42,
        instability_threshold=0.35,
        trustworthiness_threshold=0.85,
    ):
        self.n_neighbors = n_neighbors
        self.n_bootstraps = n_bootstraps
        self.sample_fraction = sample_fraction
        self.random_state = random_state
        self.instability_threshold = instability_threshold
        self.trustworthiness_threshold = trustworthiness_threshold

    @staticmethod
    def _knn_indices(X, n_neighbors):
        n_neighbors = min(n_neighbors, max(1, len(X) - 1))
        nn = NearestNeighbors(n_neighbors=n_neighbors + 1)
        nn.fit(X)
        idx = nn.kneighbors(X, return_distance=False)
        return idx[:, 1:]

    @staticmethod
    def neighborhood_overlap(idx_a, idx_b):
        overlaps = []
        for a, b in zip(idx_a, idx_b):
            set_a = set(a.tolist())
            set_b = set(b.tolist())
            union = set_a | set_b
            overlaps.append(1.0 if len(union) == 0 else len(set_a & set_b) / len(union))
        return float(np.mean(overlaps))

    @staticmethod
    def continuity_score(X_high, X_low, n_neighbors=10):
        high_idx = LatentSpaceDiagnostics._knn_indices(X_high, n_neighbors)
        low_idx = LatentSpaceDiagnostics._knn_indices(X_low, n_neighbors)
        return LatentSpaceDiagnostics.neighborhood_overlap(high_idx, low_idx)

    def embedding_stability_report(self, X, latent_builder):
        rng = np.random.default_rng(self.random_state)

        X_values = np.asarray(X, dtype=float)
        base_builder = clone(latent_builder)
        Z_base = base_builder.fit_transform(X)
        Z_values = np.asarray(Z_base, dtype=float)

        k = min(self.n_neighbors, max(1, len(X) - 1))

        trust = trustworthiness(X_values, Z_values, n_neighbors=k)
        cont = self.continuity_score(X_values, Z_values, n_neighbors=k)

        base_neighbors = self._knn_indices(Z_values, k)
        overlaps = []
        distance_correlations = []

        n = len(X)
        boot_size = max(3, int(self.sample_fraction * n))

        D_base = pairwise_distances(Z_values)
        D_base_flat = D_base[np.triu_indices_from(D_base, k=1)]

        for _ in range(self.n_bootstraps):
            boot_idx = rng.choice(np.arange(n), size=boot_size, replace=True)
            unique_idx = np.unique(boot_idx)

            if len(unique_idx) < 3:
                continue

            X_boot = X.iloc[unique_idx]

            try:
                builder = clone(latent_builder)
                builder.fit(X_boot)
                Z_boot_full = builder.transform(X)
                Z_boot_values = np.asarray(Z_boot_full, dtype=float)

                boot_neighbors = self._knn_indices(Z_boot_values, k)
                overlaps.append(self.neighborhood_overlap(base_neighbors, boot_neighbors))

                D_boot = pairwise_distances(Z_boot_values)
                D_boot_flat = D_boot[np.triu_indices_from(D_boot, k=1)]

                corr = spearmanr(D_base_flat, D_boot_flat).correlation
                if np.isfinite(corr):
                    distance_correlations.append(float(corr))

            except Exception as e:
                warnings.warn(f"Bootstrap latent diagnostic failed: {e}", RuntimeWarning)

        mean_overlap = float(np.mean(overlaps)) if len(overlaps) else np.nan
        instability = 1.0 - mean_overlap if np.isfinite(mean_overlap) else np.nan
        mean_distance_corr = (
            float(np.mean(distance_correlations))
            if len(distance_correlations)
            else np.nan
        )

        warnings_list = []

        if trust < self.trustworthiness_threshold:
            warnings_list.append("Low trustworthiness: latent space poorly preserves local neighborhoods.")

        if np.isfinite(instability) and instability > self.instability_threshold:
            warnings_list.append("High bootstrap instability: latent topology is not stable.")

        if warnings_list:
            warnings.warn(
                "Latent-space diagnostic warnings: " + " | ".join(warnings_list),
                RuntimeWarning,
            )

        return {
            "trustworthiness": float(trust),
            "continuity_overlap": float(cont),
            "bootstrap_neighbor_overlap": mean_overlap,
            "bootstrap_instability": instability,
            "bootstrap_distance_spearman": mean_distance_corr,
            "n_bootstraps_effective": int(len(overlaps)),
            "latent_reliable": bool(
                trust >= self.trustworthiness_threshold
                and (not np.isfinite(instability) or instability <= self.instability_threshold)
            ),
            "warnings": warnings_list,
        }
