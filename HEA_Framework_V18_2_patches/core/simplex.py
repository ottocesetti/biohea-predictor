# ============================================================
# FILE: core/simplex.py
# PATCHED — V18.2
# tolerant closure + multiplicative replacement
# ============================================================

import numpy as np
import pandas as pd


class SimplexUtils:
    """
    Utilities for compositional data.

    Experimental compositions often do not close exactly because of rounding,
    truncation, analytical uncertainty, or unreported trace elements.
    This module validates, closes, and applies multiplicative replacement
    before ILR.
    """

    @staticmethod
    def closure(X, eps=1e-15):
        X = np.asarray(X, dtype=float)
        row_sum = X.sum(axis=1, keepdims=True)
        if np.any(row_sum <= eps):
            raise ValueError("Invalid composition: at least one row has zero or negative total.")
        return X / row_sum

    @staticmethod
    def multiplicative_replacement(X, delta=1e-8):
        """
        CoDA-safe zero replacement.

        Replaces zeros by a small delta while rescaling positive parts
        multiplicatively, preserving relative ratios among non-zero components.
        """
        X = np.asarray(X, dtype=float)
        X = SimplexUtils.closure(X)

        X_fixed = X.copy()
        zero_mask = X_fixed <= 0
        n_zero = zero_mask.sum(axis=1, keepdims=True)

        for i in range(X_fixed.shape[0]):
            k = int(n_zero[i, 0])
            if k == 0:
                continue
            if k * delta >= 1.0:
                raise ValueError("Zero replacement delta is too large for this composition.")

            pos_mask = ~zero_mask[i]
            if X_fixed[i, pos_mask].sum() <= 0:
                raise ValueError("Invalid composition: all components are zero after closure.")

            X_fixed[i, zero_mask[i]] = delta
            X_fixed[i, pos_mask] *= (1.0 - k * delta) / X_fixed[i, pos_mask].sum()

        return SimplexUtils.closure(X_fixed)

    @staticmethod
    def prepare_composition(
        X,
        columns=None,
        closure_tolerance=1e-3,
        zero_replacement=1e-8,
        normalize=True,
        allow_zero=True,
    ):
        """
        Validates and prepares a composition DataFrame for ILR transformation.
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        X = X.copy().astype(float)

        if columns is not None:
            missing = [c for c in columns if c not in X.columns]
            if missing:
                raise ValueError(f"Missing composition columns required by fitted model: {missing}")
            X = X[columns]

        if not np.isfinite(X.values).all():
            raise ValueError("Composition contains NaN or infinite values.")

        if (X < 0).any().any():
            raise ValueError("Negative composition detected.")

        sums = X.sum(axis=1)
        if not np.allclose(sums, 1.0, atol=closure_tolerance):
            if not normalize:
                raise ValueError(
                    "Compositions do not satisfy closure. "
                    "Set normalize=True or provide closed atomic fractions."
                )

        if normalize:
            X = pd.DataFrame(
                SimplexUtils.closure(X.values),
                columns=X.columns,
                index=X.index,
            )

        if (X <= 0).any().any():
            if not allow_zero:
                raise ValueError("Zero composition detected. ILR requires strictly positive values.")

            X = pd.DataFrame(
                SimplexUtils.multiplicative_replacement(X.values, delta=zero_replacement),
                columns=X.columns,
                index=X.index,
            )

        return X
