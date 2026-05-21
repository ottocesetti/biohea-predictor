# ============================================================
# FILE: validation/interaction_discovery.py
# ============================================================

import itertools
import pandas as pd

from sklearn.feature_selection import mutual_info_regression


class SparseInteractionScreener:

    def __init__(self, max_candidates=25, random_state=42):
        self.max_candidates = max_candidates
        self.random_state = random_state

    def screen_pairwise_products(self, X, y, allowed_columns=None):
        if allowed_columns is None:
            allowed_columns = list(X.columns)

        allowed_columns = [c for c in allowed_columns if c in X.columns]
        candidates = {}

        for a, b in itertools.combinations(allowed_columns, 2):
            candidates[f"{a}_x_{b}"] = X[a].values * X[b].values

        if not candidates:
            return pd.DataFrame(columns=["candidate", "mutual_information"])

        C = pd.DataFrame(candidates, index=X.index)

        mi = mutual_info_regression(
            C,
            y,
            random_state=self.random_state,
        )

        out = pd.DataFrame({
            "candidate": C.columns,
            "mutual_information": mi,
        }).sort_values("mutual_information", ascending=False)

        return out.head(self.max_candidates)
