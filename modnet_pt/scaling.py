"""Input scaling/imputation with upstream semantics.

Port of ``MODNetModel._set_scale_impute`` (upstream vanilla.py:236-254)
plus the fit-time overrides from ``MODNetModel.fit`` (vanilla.py:388-403):
min-max to [-0.5, 0.5] with NaN → -1, or standard scaling with NaN →
10× the max scaled value; string strategies delegate to sklearn's
``SimpleImputer``. Scaling always happens before imputation (upstream
``xscale_before_impute=True`` default). See DESIGN.md §2.3 (milestone M3).
"""

from __future__ import annotations

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler


class ScaleImpute:
    """Fitted scaler+imputer with upstream MODNet semantics."""

    def __init__(self, xscale: str | None = "minmax", impute_missing=0):
        self.xscale = xscale
        self.impute_missing = impute_missing

    def fit(self, X: np.ndarray):
        X = np.asarray(X, dtype=np.float64)

        if self.xscale == "minmax":
            scaler = MinMaxScaler(feature_range=(-0.5, 0.5))
            fill = -1 if not isinstance(self.impute_missing, str) else self.impute_missing
        elif self.xscale == "standard":
            scaler = StandardScaler()
            if not isinstance(self.impute_missing, str):
                # vanilla.py:398-401: NaN → 10 × max of the scaled matrix
                fill = 10 * np.max(np.nan_to_num(scaler.fit_transform(X)))
            else:
                fill = self.impute_missing
        elif self.xscale is None:
            scaler = None
            fill = self.impute_missing
        else:
            raise ValueError(
                f"xscale must be None, 'minmax' or 'standard', got {self.xscale!r}"
            )

        if isinstance(fill, str):
            imputer = SimpleImputer(missing_values=np.nan, strategy=fill)
        else:
            imputer = SimpleImputer(
                missing_values=np.nan, strategy="constant", fill_value=fill
            )

        if scaler is not None:
            self._pipeline = Pipeline([("scaler", scaler), ("imputer", imputer)])
        else:
            self._pipeline = Pipeline([("imputer", imputer)])
        self._pipeline.fit(X)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return self._pipeline.transform(np.asarray(X, dtype=np.float64))
