# ============================================================
# FILE: core/latent_features.py
# ============================================================

import warnings
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


class LatentFeatureBuilder(BaseEstimator, TransformerMixin):

    FRAMEWORK_VERSION = "18.2"

    def __init__(
        self,
        method="umap",
        n_components=8,
        n_neighbors=25,
        min_dist=0.1,
        metric="euclidean",
        random_state=42,
        clip_range=None,
        fallback_to_pca=True,
    ):
        self.method = method
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.metric = metric
        self.random_state = random_state
        self.clip_range = clip_range
        self.fallback_to_pca = fallback_to_pca
        self.scaler = StandardScaler()
        self.reducer = None
        self.columns_ = None
        self.metadata_ = {}

    def _make_reducer(self, n_samples):
        n_components = min(int(self.n_components), max(1, n_samples - 1))

        if self.method.lower() == "pca":
            return PCA(n_components=n_components, random_state=self.random_state)

        if self.method.lower() == "umap":
            try:
                from umap import UMAP

                n_neighbors = min(int(self.n_neighbors), max(2, n_samples - 1))
                return UMAP(
                    n_components=n_components,
                    n_neighbors=n_neighbors,
                    min_dist=self.min_dist,
                    metric=self.metric,
                    random_state=self.random_state,
                    transform_seed=self.random_state,
                    low_memory=True,
                )
            except Exception as e:
                if not self.fallback_to_pca:
                    raise
                warnings.warn(
                    f"UMAP unavailable or failed during reducer creation: {e}. Falling back to PCA.",
                    RuntimeWarning,
                )
                self.method = "pca"
                return PCA(n_components=n_components, random_state=self.random_state)

        raise ValueError("method must be either 'umap' or 'pca'.")

    def fit(self, X, y=None):
        Xs = self.scaler.fit_transform(X)
        self.reducer = self._make_reducer(n_samples=Xs.shape[0])
        self.reducer.fit(Xs)

        n_out = getattr(self.reducer, "n_components", self.n_components)
        self.columns_ = [f"latent_{i}" for i in range(int(n_out))]

        self.metadata_ = {
            "framework_version": self.FRAMEWORK_VERSION,
            "method": self.method,
            "n_components_requested": self.n_components,
            "n_components_effective": int(n_out),
            "n_neighbors_requested": self.n_neighbors,
            "min_dist": self.min_dist,
            "metric": self.metric,
            "random_state": self.random_state,
            "interpretability": "non_physical_embedding",
            "fit_sample_size": int(Xs.shape[0]),
            "leakage_policy": "fit only on training folds",
        }
        return self

    def transform(self, X):
        if self.reducer is None or self.columns_ is None:
            raise RuntimeError("LatentFeatureBuilder must be fitted before transform().")

        Xs = self.scaler.transform(X)
        Z = self.reducer.transform(Xs)

        if self.clip_range is not None:
            low, high = self.clip_range
            if np.any((Z < low) | (Z > high)):
                warnings.warn(
                    "Latent embedding exceeded clip range. This may indicate extrapolation.",
                    RuntimeWarning,
                )
            Z = np.clip(Z, low, high)

        return pd.DataFrame(Z, columns=self.columns_, index=X.index)

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)
