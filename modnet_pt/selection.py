"""FRR feature selection: normalized mutual information + relevance-redundancy.

Port of upstream MODNet (v0.4.4 @ 1281977):
  modnet/preprocessing.py:42   compute_mi        (sklearn kNN MI, n_neighbors=3)
  modnet/preprocessing.py:64    nmi_target
  modnet/preprocessing.py:171   get_cross_nmi     (EPS=1e-16, drop_thr=0.2)
  modnet/preprocessing.py:293   dynamic p = max(0.1, 4.5 - 0.4*n**0.4)
  modnet/preprocessing.py:307   dynamic c = min(1e5, 1e-6 * n**3)
  modnet/preprocessing.py:476   get_features_dyn  (greedy RR ranking)
  modnet/preprocessing.py:516   merge_ranked      (multi-target union)
  modnet/sklearn.py:62          RR transformer   (finished properly here)
Parity bar: the ranking matches upstream bit-for-bit on fixed inputs (same
per-pair single-column ``mutual_info_regression`` calls, same seeds). See
DESIGN.md §2.2 (milestone M2).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.validation import check_is_fitted

EPS = 1e-16  # upstream preprocessing.py:39

DEFAULT_SUBSAMPLE_SEED = 12  # upstream MODData.feature_selection samples cross-NMI with random_state=12


def _compute_mi(x, y, n_neighbors=3, random_state=None):
    """Per-pair kNN mutual information, mirroring upstream ``compute_mi``
    (preprocessing.py:42): single-column reshape, sklearn defaults."""
    return mutual_info_regression(
        x.reshape(-1, 1), y, random_state=random_state, n_neighbors=n_neighbors
    )[0]


def _preprocess_nans(df_feat: pd.DataFrame) -> pd.DataFrame:
    """NaN-aware min-max preprocessing shared by ``nmi_target`` and
    ``get_cross_nmi`` (preprocessing.py:125-133, 205-213): min-max to
    [-0.5, 0.5] then NaN → -1, only when NaNs are present."""
    if df_feat.isna().any().any():
        scaler = MinMaxScaler(feature_range=(-0.5, 0.5))
        x = scaler.fit_transform(df_feat.values)
        x = np.nan_to_num(x, nan=-1)
        df_feat = pd.DataFrame(x, index=df_feat.index, columns=df_feat.columns)
    return df_feat


def nmi_target(
    df_feat: pd.DataFrame,
    df_target: pd.DataFrame,
    n_neighbors: int = 3,
    random_state: int | None = None,
) -> pd.DataFrame:
    """NMI between each feature and a single regression target.

    Port of upstream ``nmi_target`` (preprocessing.py:64) for regression
    tasks: duplicate/constant features are dropped before computing MI
    (they are never selectable), the MI matrix is computed in one
    multi-column ``mutual_info_regression`` call, and each MI is normalized
    by ``0.5 * (H(target) + H(feature))``.

    Returns a DataFrame with one column (the target) indexed by feature.
    """
    if df_target.shape[1] != 1:
        raise ValueError("The target DataFrame should have exactly one column.")
    if len(df_feat) != len(df_target):
        raise ValueError(
            "The input features DataFrame and the target variable DataFrame "
            "should contain the same number of data points."
        )

    # Drop duplicate then constant features (upstream order, :117-123)
    df_feat = df_feat.T.drop_duplicates().T
    df_feat = df_feat.loc[:, (df_feat != df_feat.iloc[0]).any()]

    df_feat = _preprocess_nans(df_feat)

    target_name = df_target.columns[0]
    mi_values = mutual_info_regression(
        df_feat,
        df_target[target_name],
        random_state=random_state,
        n_neighbors=n_neighbors,
    )
    # float dtype from the start: object-dtype columns break nlargest on
    # pandas > 1.5 (upstream hits the same issue and patches it with
    # .apply(pd.to_numeric) at preprocessing.py:364)
    mutual_info = pd.DataFrame(
        np.asarray(mi_values, dtype=float),
        index=df_feat.columns,
        columns=[target_name],
    )

    # Self-MI (entropy) of the target and of each feature (:149-159)
    target_mi = mutual_info_regression(
        df_target[target_name].values.reshape(-1, 1),
        df_target[target_name],
        random_state=random_state,
        n_neighbors=n_neighbors,
    )[0]
    diag = {
        x: mutual_info_regression(
            df_feat[x].values.reshape(-1, 1),
            df_feat[x],
            random_state=random_state,
            n_neighbors=n_neighbors,
        )[0]
        for x in df_feat.columns
    }

    for x in mutual_info.index:
        mutual_info.loc[x, target_name] = mutual_info.loc[x, target_name] / (
            (target_mi + diag[x]) / 2
        )

    mutual_info.fillna(0, inplace=True)
    return mutual_info


def get_cross_nmi(
    df_feat: pd.DataFrame,
    drop_thr: float = 0.2,
    n_neighbors: int = 3,
    random_state: int | None = None,
    n_jobs: int = 1,
) -> pd.DataFrame:
    """NMI between all pairs of features.

    Port of upstream ``get_cross_nmi`` (preprocessing.py:171): features with
    information entropy below ``drop_thr`` (or empty range) are dropped;
    pairwise MI is normalized by ``0.5 * (H_i + H_j)``; the diagonal is 1.

    Improvement over upstream: MI pairs are computed with joblib instead of
    a raw ``multiprocessing.Pool``, and each pair uses the identical
    single-column ``mutual_info_regression`` call, preserving bit-for-bit
    numerics for a fixed ``random_state``.
    """
    df_feat = _preprocess_nans(df_feat)

    columns = list(df_feat.columns)

    def _entropy(x_feat):
        return _compute_mi(
            df_feat[x_feat].values,
            df_feat[x_feat].values,
            n_neighbors=n_neighbors,
            random_state=random_state,
        )

    entropies = {
        feat: ent
        for feat, ent in zip(
            columns,
            Parallel(n_jobs=n_jobs)(
                delayed(_entropy)(x_feat) for x_feat in columns
            ),
        )
    }

    keep = [
        feat
        for feat in columns
        if entropies[feat] >= drop_thr
        and abs(df_feat[feat].max() - df_feat[feat].min()) > EPS
    ]

    mutual_info = pd.DataFrame(
        np.full((len(keep), len(keep)), np.nan),
        columns=keep,
        index=keep,
        dtype=float,
    )
    for feat in keep:
        mutual_info.loc[feat, feat] = 1.0

    pairs = [
        (xi, xj)
        for idx, xi in enumerate(keep)
        for xj in keep[idx + 1 :]
    ]

    def _pair(pair):
        xi, xj = pair
        mi = _compute_mi(
            df_feat[xi].values,
            df_feat[xj].values,
            n_neighbors=n_neighbors,
            random_state=random_state,
        )
        return xi, xj, mi / (0.5 * (entropies[xi] + entropies[xj]))

    for xi, xj, nmi in Parallel(n_jobs=n_jobs)(delayed(_pair)(p) for p in pairs):
        mutual_info.loc[xi, xj] = mutual_info.loc[xj, xi] = nmi

    mutual_info.fillna(0, inplace=True)
    return mutual_info


def get_rr_p_parameter_default(nn: int) -> float:
    """Default RR ``p`` from arXiv:2004.14766 (preprocessing.py:293)."""
    return max(0.1, 4.5 - 0.4 * nn**0.4)


def get_rr_c_parameter_default(nn: int) -> float:
    """Default RR ``c`` from arXiv:2004.14766 (preprocessing.py:307)."""
    return min(1e5, 1e-6 * nn**3)


def _resolve_rr_parameters(rr_parameters):
    """Resolve the ``p``/``c`` overrides (preprocessing.py:377-410):
    callables or ``{"function": "constant", "value": ...}`` dicts."""
    if rr_parameters is None:
        return get_rr_p_parameter_default, get_rr_c_parameter_default
    if "p" not in rr_parameters or "c" not in rr_parameters:
        raise ValueError(
            "When tuning p and c with rr_parameters, both parameters should be tuned"
        )
    getters = {}
    for key, default in (("p", None), ("c", None)):
        spec = rr_parameters[key]
        if callable(spec):
            getters[key] = spec
        elif isinstance(spec, dict) and spec.get("function") == "constant":
            value = spec["value"]
            getters[key] = lambda _, value=value: value
        else:
            raise ValueError(
                f'If not passing a callable, "{key}" dict must contain keys '
                '"function" and "value".'
            )
    return getters["p"], getters["c"]


def _rank_features_scored(cross_nmi, target_nmi, n_feat, rr_parameters=None):
    """Greedy relevance-redundancy ranking with per-step scores.

    Port of upstream ``get_features_dyn`` (preprocessing.py:476), extended
    to also record the RR score at which each feature was selected (the
    explainer's FRR widget replays exactly this trace) and to accept the
    ``rr_parameters`` p/c overrides from ``get_features_relevance_redundancy``
    (preprocessing.py:321).

    ``target_nmi`` is a Series (or single-column DataFrame) of feature→NMI.
    Returns ``(features, scores)``.
    """
    missing = [x for x in cross_nmi.index if x not in target_nmi.index]
    cross_nmi = cross_nmi.drop(missing, axis=0).drop(missing, axis=1)

    if isinstance(target_nmi, pd.DataFrame):
        target_nmi = target_nmi.iloc[:, 0]
    missing = [x for x in target_nmi.index if x not in cross_nmi.index]
    target_nmi = target_nmi.drop(missing)
    target_nmi = pd.Series(
        np.nan_to_num(target_nmi.values, nan=0.0, posinf=0.0, neginf=0.0),
        index=target_nmi.index,
    )

    get_p, get_c = _resolve_rr_parameters(rr_parameters)

    first_feature = target_nmi.nlargest(1).index[0]
    feature_set = [first_feature]
    scores = [None]

    if n_feat == -1:
        n_feat = len(cross_nmi.index)
    else:
        n_feat = min(len(cross_nmi.index), n_feat)

    for n in range(n_feat - 1):
        p = get_p(n)
        c = get_c(n)

        score = cross_nmi.copy()
        score = score.drop(feature_set, axis=0)
        score = score[feature_set]

        for i in score.index:
            row = score.loc[i, :]
            score.loc[i, :] = target_nmi[i] / (row**p + c)

        scores_remaining_features = score.min(axis=1)
        next_feature = scores_remaining_features.idxmax(axis=0)
        feature_set.append(next_feature)
        scores.append(scores_remaining_features[next_feature])

    return feature_set, scores


def get_features_dyn(n_feat, cross_nmi, target_nmi, rr_parameters=None):
    """Upstream-compatible ranking (preprocessing.py:476): returns the
    ranked feature list only."""
    features, _ = _rank_features_scored(cross_nmi, target_nmi, n_feat, rr_parameters)
    return features


def merge_ranked(lists) -> list:
    """Merge per-target rankings keeping each feature at its best rank.

    Port of upstream ``merge_ranked`` (preprocessing.py:516)."""
    lists = [list(sublist) for sublist in lists]
    max_len = max(len(sublist) for sublist in lists)
    for ind, sublist in enumerate(lists):
        if len(sublist) < max_len:
            sublist.extend((max_len - len(sublist)) * [None])

    total_set = set()
    ranked_list = []
    for subrank in zip(*lists):
        for feature in subrank:
            if feature not in total_set and feature is not None:
                ranked_list.append(feature)
                total_set.add(feature)
    return ranked_list


class FRFeatureSelector(TransformerMixin, BaseEstimator):
    """Greedy relevance–redundancy selector over NMI-ranked features.

    ``fit(X, y)`` computes the target NMI (``nmi_target_``), the cross-NMI
    (``cross_nmi_``) and the FRR ranking; one ranking per target column,
    merged with ``merge_ranked`` semantics (preprocessing.py:516).
    ``transform`` keeps the top ``n_features`` columns in ranked order
    (order = the upstream "optimal descriptors").

    Parameters mirror upstream defaults: ``n_neighbors=3``
    (preprocessing.py:48), ``n_samples=6000`` (feature_selection subsample
    size), ``drop_thr=0.2`` (cross-NMI entropy drop).

    Note: upstream subsamples the cross-NMI computation with a fixed seed
    (``random_state=12``) and the target NMI unseeded; here both subsamples
    use ``random_state`` (defaulting to the upstream fixed seed), so fits are
    deterministic and match upstream when ``random_state=12``.
    """

    def __init__(self, n_features: int = 64, rr_parameters: dict | None = None,
                 n_neighbors: int = 3, n_samples: int = 6000,
                 drop_thr: float = 0.2, random_state: int | None = None,
                 n_jobs: int = 1):
        self.n_features = n_features
        self.rr_parameters = rr_parameters
        self.n_neighbors = n_neighbors
        self.n_samples = n_samples
        self.drop_thr = drop_thr
        self.random_state = random_state
        self.n_jobs = n_jobs

    def _as_frame(self, X, columns=None):
        if isinstance(X, pd.DataFrame):
            return X
        if columns is not None and len(columns) == X.shape[1]:
            return pd.DataFrame(X, columns=columns)
        return pd.DataFrame(X)

    def fit(self, X, y=None):
        if y is None:
            raise ValueError("FRFeatureSelector requires a target y.")
        seed = (
            self.random_state
            if self.random_state is not None
            else DEFAULT_SUBSAMPLE_SEED
        )
        X_frame = self._as_frame(X)
        if isinstance(y, pd.Series):
            y_frame = y.to_frame()
        elif isinstance(y, pd.DataFrame):
            y_frame = y
        else:
            y_frame = pd.DataFrame(np.asarray(y))

        if len(X_frame) != len(y_frame):
            raise ValueError("X and y have different numbers of samples.")

        # Subsample for the (expensive) NMI computations, mirroring
        # MODData.feature_selection (preprocessing.py:867-902)
        if len(X_frame) > self.n_samples:
            df_nmi = X_frame.sample(n=self.n_samples, random_state=seed)
        else:
            df_nmi = X_frame
        y_nmi = y_frame.loc[df_nmi.index]

        self.nmi_target_ = pd.concat(
            [
                nmi_target(
                    df_nmi,
                    y_nmi[[target]],
                    n_neighbors=self.n_neighbors,
                    random_state=seed,
                )
                for target in y_nmi.columns
            ],
            axis=1,
        )
        self.cross_nmi_ = get_cross_nmi(
            df_nmi,
            drop_thr=self.drop_thr,
            n_neighbors=self.n_neighbors,
            random_state=seed,
            n_jobs=self.n_jobs,
        )

        rankings = {}
        traces = {}
        for target in self.nmi_target_.columns:
            features, scores = _rank_features_scored(
                self.cross_nmi_,
                self.nmi_target_[target],
                -1,
                rr_parameters=self.rr_parameters,
            )
            rankings[target] = features
            traces[target] = scores

        self.ranking_by_target_ = rankings
        self.selection_trace_ = traces
        if len(rankings) == 1:
            self.ranking_ = next(iter(rankings.values()))
        else:
            self.ranking_ = merge_ranked(list(rankings.values()))
        return self

    def transform(self, X):
        check_is_fitted(self, "ranking_")
        X_frame = self._as_frame(X, columns=getattr(self, "feature_names_in_", None))
        if self.n_features == -1:
            keep = self.ranking_
        else:
            keep = self.ranking_[: self.n_features]
        missing = [c for c in keep if c not in X_frame.columns]
        if missing:
            raise ValueError(
                f"Transform input is missing {len(missing)} ranked features, "
                f"e.g. {missing[:3]}."
            )
        return X_frame.loc[:, keep]

    def _more_tags(self):
        return {"requires_y": True, "preserves_dtype": []}
