# ============================================================
# FILE: models/applicability_domain.py
# ============================================================

import numpy as np

from sklearn.base import BaseEstimator
from sklearn.covariance import LedoitWolf
from sklearn.neighbors import LocalOutlierFactor


class ApplicabilityDomain(BaseEstimator):

    def __init__(
        self,
        max_neighbors=25,
        weights=None,
        evt_quantile=0.90,
        ood_threshold=0.95,
    ):
        self.max_neighbors = max_neighbors
        self.weights = weights
        self.evt_quantile = evt_quantile
        self.ood_threshold = ood_threshold
        self.ilr_cov = LedoitWolf()
        self.phys_cov = LedoitWolf()

    def fit(self, X_ilr, X_phys, X_latent):
        X_ilr = np.asarray(X_ilr, dtype=float)
        X_phys = np.asarray(X_phys, dtype=float)
        X_latent = np.asarray(X_latent, dtype=float)

        self.ilr_cov.fit(X_ilr)
        self.phys_cov.fit(X_phys)

        n_neighbors = min(self.max_neighbors, len(X_latent) - 1)
        n_neighbors = max(n_neighbors, 2)

        self.lof = LocalOutlierFactor(novelty=True, n_neighbors=n_neighbors)
        self.lof.fit(X_latent)

        self.ilr_ref_ = np.sort(self.ilr_cov.mahalanobis(X_ilr))
        self.phys_ref_ = np.sort(self.phys_cov.mahalanobis(X_phys))
        self.lof_ref_ = np.sort(-self.lof.score_samples(X_latent))

        self.evt_models_ = {
            "ilr": self._fit_evt(self.ilr_ref_),
            "physical": self._fit_evt(self.phys_ref_),
            "latent": self._fit_evt(self.lof_ref_),
        }

        if self.weights is None:
            self.weights_ = np.array([1 / 3, 1 / 3, 1 / 3], dtype=float)
        else:
            self.weights_ = np.asarray(self.weights, dtype=float)
            self.weights_ = self.weights_ / self.weights_.sum()

        return self

    @staticmethod
    def _percentile(scores, ref_sorted):
        scores = np.asarray(scores, dtype=float)
        return np.searchsorted(ref_sorted, scores, side="right") / len(ref_sorted)

    def _fit_evt(self, ref_sorted):
        threshold = float(np.quantile(ref_sorted, self.evt_quantile))
        excess = ref_sorted[ref_sorted > threshold] - threshold

        model = {
            "threshold": threshold,
            "available": False,
            "shape": None,
            "loc": None,
            "scale": None,
            "n_tail": int(len(excess)),
        }

        if len(excess) < 5:
            return model

        try:
            from scipy.stats import genpareto

            shape, loc, scale = genpareto.fit(excess, floc=0)
            if scale <= 0:
                return model

            model.update({
                "available": True,
                "shape": float(shape),
                "loc": float(loc),
                "scale": float(scale),
            })
        except Exception:
            pass

        return model

    @staticmethod
    def _evt_tail_probability(scores, evt_model):
        scores = np.asarray(scores, dtype=float)
        threshold = evt_model["threshold"]
        out = np.zeros_like(scores, dtype=float)
        mask = scores > threshold

        if not np.any(mask):
            return out

        excess = scores[mask] - threshold

        if not evt_model["available"]:
            out[mask] = 1.0
            return out

        try:
            from scipy.stats import genpareto

            out[mask] = genpareto.cdf(
                excess,
                evt_model["shape"],
                loc=evt_model["loc"],
                scale=evt_model["scale"],
            )
        except Exception:
            out[mask] = 1.0

        return out

    def evaluate(self, X_ilr, X_phys, X_latent):
        X_ilr = np.asarray(X_ilr, dtype=float)
        X_phys = np.asarray(X_phys, dtype=float)
        X_latent = np.asarray(X_latent, dtype=float)

        ilr_raw = self.ilr_cov.mahalanobis(X_ilr)
        phys_raw = self.phys_cov.mahalanobis(X_phys)
        lof_raw = -self.lof.score_samples(X_latent)

        ilr_pct = self._percentile(ilr_raw, self.ilr_ref_)
        phys_pct = self._percentile(phys_raw, self.phys_ref_)
        latent_pct = self._percentile(lof_raw, self.lof_ref_)

        ilr_evt = self._evt_tail_probability(ilr_raw, self.evt_models_["ilr"])
        phys_evt = self._evt_tail_probability(phys_raw, self.evt_models_["physical"])
        latent_evt = self._evt_tail_probability(lof_raw, self.evt_models_["latent"])

        percentile_matrix = np.column_stack([ilr_pct, phys_pct, latent_pct])
        evt_matrix = np.column_stack([ilr_evt, phys_evt, latent_evt])

        combined_percentile_risk = percentile_matrix @ self.weights_
        combined_evt_tail_risk = evt_matrix @ self.weights_

        return {
            "ilr_mahalanobis": ilr_raw,
            "physical_mahalanobis": phys_raw,
            "latent_lof": lof_raw,
            "ilr_risk_percentile": ilr_pct,
            "physical_risk_percentile": phys_pct,
            "latent_risk_percentile": latent_pct,
            "ilr_evt_tail_risk": ilr_evt,
            "physical_evt_tail_risk": phys_evt,
            "latent_evt_tail_risk": latent_evt,
            "combined_percentile_risk": combined_percentile_risk,
            "combined_evt_tail_risk": combined_evt_tail_risk,
            "out_of_domain_flag": combined_percentile_risk >= self.ood_threshold,
        }
