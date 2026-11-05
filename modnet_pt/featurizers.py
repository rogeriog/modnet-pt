"""Featurization transformer wrapping matminer.

Port target (upstream v0.4.4 @ 1281977):
  modnet/featurizers/featurizers.py:17-293  (MODFeaturizer)
  modnet/featurizers/presets/__init__.py:13  (six presets)
Upstream feeds featurized dataframes through a MODData container; here the
transformer owns fit-state (all-NaN / duplicate column drops on the training
set only) so Pipelines stay leak-free. See DESIGN.md §2.1 (milestone M1).
"""

from __future__ import annotations

from sklearn.base import BaseEstimator, TransformerMixin

PRESETS = (
    "composition_only_matminer_2023",
    "composition_only_matminer_all_2023",
    "composition_only_debreuck_2020",
    "matminer_2023",
    "matminer_all_2023",
    "debreuck_2020",
)


class MatminerFeaturizer(TransformerMixin, BaseEstimator):
    """Turn pymatgen compositions/structures into matminer descriptor columns.

    Parameters mirror the upstream preset registry; ``preset`` selects which
    featurizer set runs. Fit learns the dropped-column list; transform applies
    it. Implementation: milestone M1.
    """

    def __init__(self, preset: str = "matminer_2023", drop_allnan: bool = True,
                 drop_duplicates: bool = True):
        self.preset = preset
        self.drop_allnan = drop_allnan
        self.drop_duplicates = drop_duplicates

    def fit(self, X, y=None):
        if self.preset not in PRESETS:
            raise ValueError(f"unknown preset {self.preset!r}; choose from {PRESETS}")
        raise NotImplementedError("milestone M1 — see DESIGN.md §2.1")

    def transform(self, X):
        raise NotImplementedError("milestone M1 — see DESIGN.md §2.1")
