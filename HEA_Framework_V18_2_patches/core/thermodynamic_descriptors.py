# ============================================================
# FILE: core/thermodynamic_descriptors.py
# PATCHED — V18.2
# uppercase convention + safe fingerprint cache + masked missingness
# ============================================================

import json
import hashlib
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin


class ThermodynamicDescriptorBuilder(BaseEstimator, TransformerMixin):
    """
    CALPHAD-derived descriptors.

    Canonical scalar convention:
        GM, HM, SM

    Semantics:
        NaN   = solver or thermodynamic failure before imputation
        0.0   = physical phase absence
        value = valid thermodynamic quantity

    Missingness masks are included:
        GM_missing_flag, HM_missing_flag, SM_missing_flag
    """

    FAILURE_TYPES = [
        "timeout",
        "solver_convergence",
        "infeasible_equilibrium",
        "license_or_import",
        "unknown_failure",
    ]

    SCALAR_COLUMNS = ["GM", "HM", "SM"]

    def __init__(self, engine, impute=True):
        self.engine = engine
        self.impute = impute
        self.columns_ = None
        self.impute_values_ = None
        self._fit_cache_ = None
        self._fit_fingerprint_ = None

    def _fingerprint(self, X):
        Xn = X.copy().astype(float)
        records = []
        for _, row in Xn.iterrows():
            records.append({k: round(float(v), 12) for k, v in sorted(row.to_dict().items())})
        payload = json.dumps(records, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()

    def _single_calculation(self, composition):
        result = self.engine.calculate(composition)
        status = result.get("status", "failure")
        failure_type = result.get("failure_type", "unknown_failure")

        out = {
            "GM": result.get("GM", np.nan),
            "HM": result.get("HM", np.nan),
            "SM": result.get("SM", np.nan),
            "thermo_success_flag": int(status == "success"),
            "thermo_failure_flag": int(status != "success"),
            "thermo_metastable_flag": int(getattr(self.engine, "metastable", False)),
            "thermo_suspended_phases_count": len(getattr(self.engine, "suspended_phases", [])),
        }

        for col in self.SCALAR_COLUMNS:
            out[f"{col}_missing_flag"] = int(pd.isna(out[col]))

        for ft in self.FAILURE_TYPES:
            out[f"thermo_failure_{ft}"] = int(failure_type == ft)

        phases = result.get("phase_fractions", {})
        for phase, frac in phases.items():
            out[f"phase_{phase}"] = float(frac)

        return out

    def _compute_raw(self, X):
        rows = []

        for _, row in X.iterrows():
            comp = row.to_dict()
            try:
                rows.append(self._single_calculation(comp))
            except Exception:
                out = {
                    "GM": np.nan,
                    "HM": np.nan,
                    "SM": np.nan,
                    "thermo_success_flag": 0,
                    "thermo_failure_flag": 1,
                    "thermo_metastable_flag": int(getattr(self.engine, "metastable", False)),
                    "thermo_suspended_phases_count": len(getattr(self.engine, "suspended_phases", [])),
                }

                for col in self.SCALAR_COLUMNS:
                    out[f"{col}_missing_flag"] = 1

                for ft in self.FAILURE_TYPES:
                    out[f"thermo_failure_{ft}"] = int(ft == "unknown_failure")

                rows.append(out)

        df = pd.DataFrame(rows, index=X.index)
        return df.replace([np.inf, -np.inf], np.nan)

    def _align_columns(self, df):
        for col in self.columns_:
            if col not in df.columns:
                if col.startswith("phase_"):
                    df[col] = 0.0
                elif col.endswith("_flag") or col.startswith("thermo_failure_"):
                    df[col] = 0.0
                else:
                    df[col] = np.nan
        return df[self.columns_]

    def _fit_imputer(self, df):
        values = {}
        for col in df.columns:
            if col.startswith("phase_"):
                values[col] = 0.0
            elif col.endswith("_flag") or col.startswith("thermo_failure_"):
                values[col] = 0.0
            else:
                median = df[col].median(skipna=True)
                values[col] = 0.0 if pd.isna(median) else float(median)
        self.impute_values_ = values

    def _apply_imputer(self, df):
        if not self.impute:
            return df
        return df.fillna(self.impute_values_)

    def fit(self, X, y=None):
        fp = self._fingerprint(X)
        df = self._compute_raw(X)
        self.columns_ = df.columns.tolist()
        self._fit_imputer(df)
        df = self._apply_imputer(df)
        self._fit_cache_ = df.copy()
        self._fit_fingerprint_ = fp
        return self

    def transform(self, X):
        if self.columns_ is None:
            raise RuntimeError("ThermodynamicDescriptorBuilder must be fitted before transform().")

        fp = self._fingerprint(X)
        if self._fit_cache_ is not None and fp == self._fit_fingerprint_:
            cached = self._fit_cache_.copy()
            cached.index = X.index
            return cached

        df = self._compute_raw(X)
        df = self._align_columns(df)
        return self._apply_imputer(df)

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)
