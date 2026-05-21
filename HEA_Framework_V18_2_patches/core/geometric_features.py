# ============================================================
# FILE: core/geometric_features.py
# ============================================================

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from core.ilr import ILRTransformer
from core.simplex import SimplexUtils


class GeometricFeatureBuilder(BaseEstimator, TransformerMixin):

    def __init__(
        self,
        closure_tolerance=1e-3,
        zero_replacement=1e-8,
        normalize=True,
        allow_zero=True,
    ):
        self.closure_tolerance = closure_tolerance
        self.zero_replacement = zero_replacement
        self.normalize = normalize
        self.allow_zero = allow_zero
        self.ilr = ILRTransformer()
        self.composition_columns_ = None

    def fit(self, X, y=None):
        Xp = SimplexUtils.prepare_composition(
            X,
            columns=None,
            closure_tolerance=self.closure_tolerance,
            zero_replacement=self.zero_replacement,
            normalize=self.normalize,
            allow_zero=self.allow_zero,
        )
        self.composition_columns_ = Xp.columns.tolist()
        self.ilr.fit(Xp.values)
        return self

    def transform(self, X):
        if self.composition_columns_ is None:
            raise RuntimeError("GeometricFeatureBuilder must be fitted before transform().")

        Xp = SimplexUtils.prepare_composition(
            X,
            columns=self.composition_columns_,
            closure_tolerance=self.closure_tolerance,
            zero_replacement=self.zero_replacement,
            normalize=self.normalize,
            allow_zero=self.allow_zero,
        )

        Z = self.ilr.transform(Xp.values)
        cols = [f"ilr_{i}" for i in range(Z.shape[1])]
        return pd.DataFrame(Z, columns=cols, index=Xp.index)

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)
