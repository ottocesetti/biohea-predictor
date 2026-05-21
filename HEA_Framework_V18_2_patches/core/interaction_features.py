# ============================================================
# FILE: core/interaction_features.py
# ============================================================

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class PhysicsInteractionGeneratorV2(BaseEstimator, TransformerMixin):

    def __init__(self, enable_pairwise=False):
        self.enable_pairwise = enable_pairwise
        self.columns_ = None
        self.metadata_ = {
            "version": "18.2",
            "physical_assumptions": [
                "delta-Smix coupling",
                "enthalpy-entropy competition",
                "VEC-radius coupling",
                "Omega stabilization",
                "thermal stability coupled to enthalpy",
            ],
            "leakage_safe": True,
            "causal_claim": False,
        }

    def fit(self, X, y=None):
        out = self._generate(X)
        self.columns_ = out.columns.tolist()
        return self

    def transform(self, X):
        if self.columns_ is None:
            raise RuntimeError("PhysicsInteractionGeneratorV2 must be fitted before transform().")

        out = self._generate(X)
        for col in self.columns_:
            if col not in out.columns:
                out[col] = 0.0
        return out[self.columns_]

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

    def _generate(self, X):
        out = pd.DataFrame(index=X.index)

        if "delta" in X.columns and "Smix" in X.columns:
            out["delta_x_smix"] = X["delta"] * X["Smix"]

        if "Hmix" in X.columns and "Smix" in X.columns:
            out["Hmix_x_Smix"] = X["Hmix"] * X["Smix"]
            out["Hmix_over_Smix"] = X["Hmix"] / (X["Smix"] + 1e-12)

        if "Omega" in X.columns and "delta" in X.columns:
            out["Omega_x_delta"] = X["Omega"] * X["delta"]

        if "VEC" in X.columns and "delta" in X.columns:
            out["VEC_x_delta"] = X["VEC"] * X["delta"]

        if "Tm" in X.columns and "Hmix" in X.columns:
            out["Tm_x_Hmix"] = X["Tm"] * X["Hmix"]

        return out
