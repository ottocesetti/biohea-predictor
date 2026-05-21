# ============================================================
# FILE: framework/HEAFrameworkV18_2.py
# PATCHED — V18.2
# framework wrapper with geometry-aware AD
# ============================================================

from sklearn.base import BaseEstimator

from core.feature_pipeline import HEAFeaturePipelineV18_2
from models.compositional_phase_model import CompositionalPhaseModel
from models.applicability_domain import ApplicabilityDomain


class HEAFrameworkV18_2(BaseEstimator):
    """
    HEA framework wrapper.

    The predictive model receives all features.
    Applicability domain receives separated geometries only.
    """

    def __init__(
        self,
        element_data,
        thermocalc_engine,
        binary_mixing_enthalpies,
        latent_method="umap",
        latent_dimensions=8,
    ):
        self.element_data = element_data
        self.thermocalc_engine = thermocalc_engine
        self.binary_mixing_enthalpies = binary_mixing_enthalpies
        self.latent_method = latent_method
        self.latent_dimensions = latent_dimensions

        self.pipeline = HEAFeaturePipelineV18_2(
            element_data=element_data,
            thermocalc_engine=thermocalc_engine,
            binary_mixing_enthalpies=binary_mixing_enthalpies,
            latent_method=latent_method,
            latent_dimensions=latent_dimensions,
        )

        self.phase_model = CompositionalPhaseModel()
        self.applicability = ApplicabilityDomain()

    @staticmethod
    def _cols(X_features, prefix=None, names=None):
        if prefix is not None:
            return [c for c in X_features.columns if c.startswith(prefix)]
        if names is not None:
            return [c for c in names if c in X_features.columns]
        return []

    def fit(self, X, y):
        X_features = self.pipeline.fit_transform(X)

        self.phase_model.fit(X_features.values, y)

        geo_cols = self._cols(X_features, prefix="ilr_")
        phys_cols = self._cols(
            X_features,
            names=[
                "delta",
                "Omega",
                "Smix",
                "Hmix",
                "VEC",
                "Tm",
                "Density",
                "CohesiveEnergy",
            ],
        )
        latent_cols = self._cols(X_features, prefix="latent_")

        self.applicability.fit(
            X_features[geo_cols].values,
            X_features[phys_cols].values,
            X_features[latent_cols].values,
        )

        self.geo_cols_ = geo_cols
        self.phys_cols_ = phys_cols
        self.latent_cols_ = latent_cols

        return self

    def predict(self, X):
        X_features = self.pipeline.transform(X)
        preds = self.phase_model.predict(X_features.values)

        applicability = self.applicability.evaluate(
            X_features[self.geo_cols_].values,
            X_features[self.phys_cols_].values,
            X_features[self.latent_cols_].values,
        )

        return {
            "phase_prediction": preds,
            "applicability": applicability,
        }
