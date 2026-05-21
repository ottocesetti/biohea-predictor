# ============================================================
# FILE: validation/thermo_missingness.py
# NEW — V18.2
# thermodynamic missingness and failure diagnostics
# ============================================================

import numpy as np
import pandas as pd


class ThermodynamicMissingnessDiagnostics:
    """
    Audits CALPHAD-derived feature missingness.

    Purpose:
    - detect systematic solver failures;
    - detect target-correlated missingness;
    - quantify whether imputed thermodynamic descriptors may become shortcuts.
    """

    def missingness_report(self, X_thermo, y=None):
        rows = []
        flag_cols = [
            c for c in X_thermo.columns
            if c.endswith("_missing_flag")
            or c.startswith("thermo_failure_")
            or c in ["thermo_failure_flag", "thermo_success_flag"]
        ]

        for col in flag_cols:
            arr = X_thermo[col].astype(float).values

            row = {
                "feature": col,
                "rate": float(np.mean(arr)),
                "count": int(np.sum(arr)),
                "n": int(len(arr)),
            }

            if y is not None:
                yy = np.asarray(y)
                if len(np.unique(arr)) > 1:
                    try:
                        row["target_mean_when_flag_1"] = float(np.mean(yy[arr == 1]))
                        row["target_mean_when_flag_0"] = float(np.mean(yy[arr == 0]))
                        row["target_shift"] = (
                            row["target_mean_when_flag_1"]
                            - row["target_mean_when_flag_0"]
                        )
                    except Exception:
                        row["target_shift"] = np.nan
                else:
                    row["target_shift"] = 0.0

            rows.append(row)

        return pd.DataFrame(rows).sort_values("rate", ascending=False)

    def systematic_failure_detected(self, X_thermo, threshold=0.10):
        if "thermo_failure_flag" not in X_thermo.columns:
            return {
                "available": False,
                "reason": "thermo_failure_flag not present.",
            }

        rate = float(X_thermo["thermo_failure_flag"].mean())

        return {
            "available": True,
            "failure_rate": rate,
            "systematic_failure": bool(rate >= threshold),
        }
