# ============================================================
# FILE: validation/feature_diagnostics.py
# ============================================================

import warnings
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_regression
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler


class FeatureDiagnostics(BaseEstimator):

    FRAMEWORK_VERSION = "18.2"

    def __init__(
        self,
        vif_threshold=10.0,
        corr_threshold=0.95,
        latent_dominance_threshold=0.50,
    ):
        self.vif_threshold = vif_threshold
        self.corr_threshold = corr_threshold
        self.latent_dominance_threshold = latent_dominance_threshold

    def correlation_report(self, X):
        corr = X.corr().abs()
        redundant_pairs = []

        for i in range(len(corr.columns)):
            for j in range(i + 1, len(corr.columns)):
                val = corr.iloc[i, j]
                if val >= self.corr_threshold:
                    redundant_pairs.append({
                        "feature_a": corr.columns[i],
                        "feature_b": corr.columns[j],
                        "correlation": float(val),
                    })

        return pd.DataFrame(redundant_pairs)

    def vif_report(self, X):
        X = X.copy().replace([np.inf, -np.inf], np.nan).fillna(0.0)

        if X.shape[1] < 2:
            return pd.DataFrame({
                "feature": X.columns,
                "VIF": [1.0] * X.shape[1],
                "high_collinearity": [False] * X.shape[1],
            })

        Xn = StandardScaler().fit_transform(X)
        rows = []

        for i in range(X.shape[1]):
            y = Xn[:, i]
            Xi = np.delete(Xn, i, axis=1)

            if np.std(y) < 1e-12:
                vif = np.inf
            else:
                model = LinearRegression()
                model.fit(Xi, y)
                pred = model.predict(Xi)
                r2 = r2_score(y, pred)
                vif = 1.0 / (1.0 - r2 + 1e-12)

            rows.append({
                "feature": X.columns[i],
                "VIF": float(vif),
                "high_collinearity": bool(vif > self.vif_threshold),
            })

        return pd.DataFrame(rows)

    def pca_report(self, X, variance_threshold=0.95):
        X = X.copy().replace([np.inf, -np.inf], np.nan).fillna(0.0)
        Xs = StandardScaler().fit_transform(X)

        pca = PCA()
        pca.fit(Xs)

        cumulative = np.cumsum(pca.explained_variance_ratio_)
        n_required = int(np.searchsorted(cumulative, variance_threshold) + 1)
        redundancy_ratio = 1.0 - (n_required / X.shape[1])
        collapse_ratio = float(pca.explained_variance_ratio_[0])

        if collapse_ratio > 0.85:
            warnings.warn("Strong PCA feature collapse detected.", RuntimeWarning)

        return {
            "n_original_features": int(X.shape[1]),
            "n_components_for_threshold": n_required,
            "variance_threshold": float(variance_threshold),
            "redundancy_ratio": float(redundancy_ratio),
            "collapse_ratio": collapse_ratio,
            "explained_variance": cumulative.tolist(),
        }

    def mutual_information_report(self, X, y):
        X = X.copy().replace([np.inf, -np.inf], np.nan).fillna(0.0)
        mi = mutual_info_regression(X, y, random_state=42)
        return pd.DataFrame({
            "feature": X.columns,
            "mutual_information": mi,
        }).sort_values("mutual_information", ascending=False)

    def permutation_report(
        self,
        model,
        X,
        y,
        n_repeats=10,
        scoring="neg_mean_absolute_error",
    ):
        result = permutation_importance(
            model,
            X,
            y,
            n_repeats=n_repeats,
            scoring=scoring,
            random_state=42,
        )

        return pd.DataFrame({
            "feature": X.columns,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }).sort_values("importance_mean", ascending=False)

    def shap_report(self, model, X_background, X_eval):
        try:
            import shap
        except Exception as e:
            raise ImportError("shap is required for shap_report(). Install shap first.") from e

        explainer = shap.Explainer(model.predict, X_background)
        values = explainer(X_eval)
        mean_abs = np.abs(values.values).mean(axis=0)

        return pd.DataFrame({
            "feature": X_eval.columns,
            "mean_abs_shap": mean_abs,
        }).sort_values("mean_abs_shap", ascending=False)

    def latent_dominance_report(self, importance_df, importance_col):
        latent_mask = importance_df["feature"].str.startswith("latent_")
        latent_importance = importance_df.loc[latent_mask, importance_col].sum()
        total_importance = importance_df[importance_col].sum()
        ratio = latent_importance / (total_importance + 1e-12)
        detected = ratio > self.latent_dominance_threshold

        if detected:
            warnings.warn(
                "Latent embeddings dominate model explanations. "
                "Potential shortcut, manifold leakage, or redundancy detected.",
                RuntimeWarning,
            )

        return {
            "latent_importance_ratio": float(ratio),
            "dominance_detected": bool(detected),
        }

    def latent_physics_leakage_report(self, X):
        latent_cols = [c for c in X.columns if c.startswith("latent_")]
        physical_cols = [c for c in X.columns if not c.startswith("latent_")]

        if len(latent_cols) == 0 or len(physical_cols) == 0:
            return {
                "available": False,
                "reason": "latent or physical columns not present.",
            }

        leakage_scores = []

        for lat in latent_cols:
            corr = X[physical_cols].corrwith(X[lat]).abs()
            leakage_scores.append(float(corr.max()))

        max_leakage = float(np.max(leakage_scores))

        if max_leakage > 0.95:
            warnings.warn("Potential shortcut between latent and physical features detected.", RuntimeWarning)

        return {
            "available": True,
            "max_cross_correlation": max_leakage,
            "potential_shortcut": bool(max_leakage > 0.95),
        }
