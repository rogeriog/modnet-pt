"""Bootstrap ensembles and MC-dropout uncertainty.

Port targets (upstream v0.4.4 @ 1281977):
  modnet/models/ensemble.py:26   EnsembleMODNetModel — n_models bootstrap
                                 fits, predict = mean, std = epistemic
                                 uncertainty (De Breuck 2021, pp. 4-5)
  modnet/models/bayesian.py:290  stochastic-prediction uncertainty loop
                                 (upstream is a deprecated DenseVariational
                                 BNN; DESIGN.md §2.3 specifies the modern
                                 MC-dropout equivalent instead)

Deviations, documented: upstream resamples every bootstrap member with the
same ``random_state=2943`` (ensemble.py:92-104), so all members see
identical data — here each member gets its own seed derived from
``random_state``. Members are ordinary fitted estimators, so cloning,
save/load and Pipeline composition keep working.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone


def fit_bootstrap_members(template, X: pd.DataFrame, y: pd.DataFrame,
                          n_models: int, seed: int):
    """Fit ``n_models`` bootstrap-resampled clones of ``template``.

    ``template`` must be unfitted with ``bootstrap=0``/``search="none"``
    semantics (callers reset those params). Resampling uses
    ``np.random.default_rng(seed)`` — per-member streams, unlike upstream's
    single fixed seed.
    """
    members = []
    rng = np.random.default_rng(seed)
    n = len(X)
    for _ in range(n_models):
        idx = rng.choice(n, size=n, replace=True)
        member = clone(template)
        member.fit(X.iloc[idx], y.iloc[idx])
        members.append(member)
    return members


def ensemble_predict(members, X):
    """Mean of member ``predict`` outputs -> DataFrame (regression;
    classification labels go through :func:`ensemble_predict_proba`)."""
    frames = [m.predict(X) for m in members]
    arr = np.stack([f.to_numpy() for f in frames])
    return pd.DataFrame(arr.mean(axis=0), columns=frames[0].columns)


def ensemble_predict_std(members, X):
    """Std of member predictions (``np.std`` ddof=0, upstream
    ensemble.py:178-182) -> DataFrame."""
    frames = [m.predict(X) for m in members]
    arr = np.stack([f.to_numpy() for f in frames])
    return pd.DataFrame(arr.std(axis=0), columns=frames[0].columns)


def ensemble_predict_proba(members, X):
    """Mean class probabilities across classifier members -> DataFrame
    with the same (target, class) MultiIndex columns."""
    frames = [m.predict_proba(X) for m in members]
    arr = np.stack([f.to_numpy() for f in frames])
    return pd.DataFrame(arr.mean(axis=0), columns=frames[0].columns)


def ensemble_predict_proba_std(members, X):
    frames = [m.predict_proba(X) for m in members]
    arr = np.stack([f.to_numpy() for f in frames])
    return pd.DataFrame(arr.std(axis=0), columns=frames[0].columns)
