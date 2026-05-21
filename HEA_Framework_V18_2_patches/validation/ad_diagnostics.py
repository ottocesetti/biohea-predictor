# ============================================================
# FILE: validation/ad_diagnostics.py
# ============================================================

import numpy as np
import pandas as pd


class ApplicabilityDomainDiagnostics:

    def score_summary(self, ad_output):
        rows = []
        for key, value in ad_output.items():
            arr = np.asarray(value)

            if arr.dtype == bool:
                rows.append({
                    "score": key,
                    "type": "boolean",
                    "mean": float(arr.mean()),
                    "std": np.nan,
                    "min": int(arr.min()),
                    "max": int(arr.max()),
                })
                continue

            if not np.issubdtype(arr.dtype, np.number):
                continue

            rows.append({
                "score": key,
                "type": "numeric",
                "mean": float(np.nanmean(arr)),
                "std": float(np.nanstd(arr)),
                "min": float(np.nanmin(arr)),
                "p50": float(np.nanpercentile(arr, 50)),
                "p95": float(np.nanpercentile(arr, 95)),
                "max": float(np.nanmax(arr)),
            })

        return pd.DataFrame(rows)

    def dominance_report(self, ad_output):
        required = [
            "ilr_risk_percentile",
            "physical_risk_percentile",
            "latent_risk_percentile",
        ]

        if not all(k in ad_output for k in required):
            return {
                "available": False,
                "reason": "Required percentile risk keys not found.",
            }

        M = np.column_stack([
            np.asarray(ad_output["ilr_risk_percentile"], dtype=float),
            np.asarray(ad_output["physical_risk_percentile"], dtype=float),
            np.asarray(ad_output["latent_risk_percentile"], dtype=float),
        ])

        labels = ["ilr", "physical", "latent"]
        winner = np.argmax(M, axis=1)

        counts = {labels[i]: int((winner == i).sum()) for i in range(3)}
        ratios = {k: v / len(winner) for k, v in counts.items()}
        dominant_space = max(ratios, key=ratios.get)

        return {
            "available": True,
            "dominant_space": dominant_space,
            "dominance_ratio": ratios[dominant_space],
            "counts": counts,
            "ratios": ratios,
        }
