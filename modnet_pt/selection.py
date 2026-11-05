"""FRR feature selection: normalized mutual information + relevance-redundancy.

Port target (upstream v0.4.4 @ 1281977):
  modnet/preprocessing.py:45   compute_mi        (sklearn kNN MI, n_neighbors=3)
  modnet/preprocessing.py:68   nmi_target
  modnet/preprocessing.py:175  get_cross_nmi
  modnet/preprocessing.py:297  dynamic p = max(0.1, 4.5 - n**0.4)
  modnet/preprocessing.py:311  dynamic c = 1e-6 * n**3
  modnet/preprocessing.py:325  get_features_relevance_redundancy (greedy RR ranking)
  modnet/preprocessing.py:476  get_features_dyn
  modnet/preprocessing.py:516  merge_ranked (multi-target union)
  modnet/sklearn.py:62         RR transformer (upstream's finished sklearn class)
Upstream starts the sklearn contract here; this module is that contract
finished properly. See DESIGN.md §2.2 (milestone M2). Parity bar: the ranking
must match upstream bit-for-bit on fixed inputs.
"""

from __future__ import annotations

from sklearn.base import BaseEstimator, TransformerMixin


class FRFeatureSelector(TransformerMixin, BaseEstimator):
    """Greedy relevance–redundancy selector over NMI-ranked features.

    fit computes target NMI, cross-NMI, and the FRR ranking; transform keeps
    the top ``n_features`` columns in ranked order (order = "optimal
    descriptors"). Exposes ``ranking_``, ``nmi_target_``, ``cross_nmi_`` and
    ``selection_trace_`` (per-step RR scores). Implementation: milestone M2.
    """

    def __init__(self, n_features: int = 64, rr_parameters: dict | None = None,
                 n_neighbors: int = 3, n_samples: int = 6000,
                 drop_thr: float = 0.2, random_state: int | None = None):
        self.n_features = n_features
        self.rr_parameters = rr_parameters
        self.n_neighbors = n_neighbors
        self.n_samples = n_samples
        self.drop_thr = drop_thr
        self.random_state = random_state

    def fit(self, X, y=None):
        raise NotImplementedError("milestone M2 — see DESIGN.md §2.2")

    def transform(self, X):
        raise NotImplementedError("milestone M2 — see DESIGN.md §2.2")
