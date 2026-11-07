"""Validation utilities: learning curves, nested CV, MatBench harness.

Port target (upstream v0.4.4 @ 1281977): modnet/matbench/benchmark.py.
Protocol per De Breuck 2021 benchmarking paper p. 6: outer 5-fold CV for
generalization error, inner 5-fold for selection/hyperparameters — which
maps directly onto ``search="preset"``/``"ga"`` estimators whose inner CV
already runs ``search_folds``/``nested`` folds.

Deviations: upstream ``matbench_benchmark`` re-runs feature selection
inside every fold on MODData; here the estimator is expected to sit inside
a ``Pipeline`` (featurize → select → model), so per-fold selection is the
pipeline's job when users wrap it — ``nested_cv``/``matbench_cv`` clone the
whole pipeline per fold.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, roc_auc_score
from sklearn.model_selection import KFold

from modnet_pt.search import kfold_splits


def _score(estimator, X_val, y_val) -> float:
    """Held-out score: MAE for regressors, negative macro-OvR AUC for
    classifiers (same convention as the search fitness functions)."""
    from sklearn.base import ClassifierMixin

    if isinstance(estimator, ClassifierMixin):
        proba = estimator.predict_proba(X_val)
        multi_label = getattr(estimator, "multi_label", False)
        aucs = []
        for col in estimator.target_names_:
            classes = estimator.classes_[col]
            labels = np.asarray(y_val[col])
            if multi_label:
                # one P(positive) column per target
                positive = classes[1]
                yt = (labels == positive).astype(int)
                if yt.sum() in (0, len(yt)):
                    continue
                p = proba[(col, positive)].to_numpy().ravel()
                aucs.append(roc_auc_score(yt, p))
                continue
            vals = []
            for cls in classes:
                yt = (labels == cls).astype(int)
                if yt.sum() in (0, len(yt)):
                    continue
                p = proba[(col, cls)].to_numpy().ravel()
                vals.append(roc_auc_score(yt, p))
            if vals:
                aucs.append(float(np.mean(vals)))
        return float(-np.mean(aucs)) if aucs else np.inf
    pred = estimator.predict(X_val)
    cols = list(pred.columns)
    return float(
        np.mean(
            [
                mean_absolute_error(
                    np.asarray(y_val[c]), pred[c].to_numpy()
                )
                for c in cols
            ]
        )
    )


def learning_curve(estimator, X, y, sizes, cv: int = 5,
                   random_state: int = 0):
    """Learning curve: mean held-out score vs training-set size.

    For each size and each CV fold, fits a clone on a seeded subsample of
    the fold's training part and scores on the held-out part. Returns a
    DataFrame with columns ``n_train_samples``, ``score_mean``,
    ``score_std`` (ddof=1 across folds), ``score_min``, ``score_max``.
    """
    X = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
    y = pd.DataFrame(y) if not isinstance(y, pd.DataFrame) else y
    sizes = sorted({int(s) for s in sizes})
    kf = KFold(n_splits=cv, shuffle=True, random_state=random_state)
    rows = []
    for size in sizes:
        fold_scores = []
        for train_idx, val_idx in kf.split(X):
            rng = np.random.default_rng(random_state + size)
            take = rng.choice(
                train_idx, size=min(size, len(train_idx)), replace=False
            )
            est = clone(estimator)
            est.fit(X.iloc[take], y.iloc[take])
            fold_scores.append(_score(est, X.iloc[val_idx], y.iloc[val_idx]))
        rows.append(
            {
                "n_train_samples": size,
                "score_mean": float(np.mean(fold_scores)),
                "score_std": float(np.std(fold_scores, ddof=1))
                if len(fold_scores) > 1
                else 0.0,
                "score_min": float(np.min(fold_scores)),
                "score_max": float(np.max(fold_scores)),
            }
        )
    return pd.DataFrame(rows)


def nested_cv(estimator, X, y, outer: int = 5, inner: int = 5,
              random_state: int = 0):
    """Nested cross-validation (De Breuck 2021 p. 6 protocol): outer folds
    measure generalization; each outer-train fit runs the estimator's own
    inner search with ``inner`` folds.

    ``estimator`` must be a *search* estimator (``search="preset"`` or
    ``"ga"``) — its ``search_folds``/``nested`` are overridden to ``inner``.
    Returns a DataFrame with per-fold scores plus ``score_mean``/``score_std``
    summary rows is NOT included — inspect ``score`` column directly.
    """
    X = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
    y = pd.DataFrame(y) if not isinstance(y, pd.DataFrame) else y
    if getattr(estimator, "search", "none") == "none":
        raise ValueError(
            "nested_cv needs a search estimator (search='preset'/'ga') so "
            "the inner loop performs model selection"
        )
    # inner folds: preset search uses search_folds, GA uses nested
    param = "search_folds" if estimator.search == "preset" else "nested"
    outer_splits = kfold_splits(
        len(X),
        y_first=y.iloc[:, 0].values
        if _is_classifier(estimator)
        else None,
        n_splits=outer,
        classification=_is_classifier(estimator),
        seed=random_state,
    )
    rows = []
    for fold, (train_idx, val_idx) in enumerate(outer_splits):
        est = clone(estimator)
        est.set_params(**{param: inner})
        est.fit(X.iloc[train_idx], y.iloc[train_idx])
        rows.append(
            {
                "fold": fold,
                "score": _score(est, X.iloc[val_idx], y.iloc[val_idx]),
            }
        )
    df = pd.DataFrame(rows)
    df.attrs["score_mean"] = float(df["score"].mean())
    df.attrs["score_std"] = float(df["score"].std(ddof=1))
    return df


def _is_classifier(estimator) -> bool:
    from sklearn.base import ClassifierMixin

    return isinstance(estimator, ClassifierMixin)


def matbench_cv(estimator, X, y, n_splits: int = 5, seed: int | None = None):
    """MatBench-style outer CV with the fixed ``MATBENCH_SEED`` splits
    (benchmark.py:16-42): clone the (search) estimator per fold, fit on the
    fold's training part, score on the held-out part. Returns per-fold
    scores DataFrame with ``score_mean``/``score_std`` attrs — the
    generalization-error protocol of the 2021 benchmarking paper.

    For full upstream parity wrap the estimator in a Pipeline with
    ``MatminerFeaturizer`` + ``FRFeatureSelector`` so feature selection is
    re-run inside each fold.
    """
    X = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
    y = pd.DataFrame(y) if not isinstance(y, pd.DataFrame) else y
    from modnet_pt.search import MATBENCH_SEED

    splits = kfold_splits(
        len(X),
        y_first=y.iloc[:, 0].values if _is_classifier(estimator) else None,
        n_splits=n_splits,
        classification=_is_classifier(estimator),
        seed=MATBENCH_SEED if seed is None else seed,
    )
    rows = []
    for fold, (train_idx, val_idx) in enumerate(splits):
        est = clone(estimator)
        est.fit(X.iloc[train_idx], y.iloc[train_idx])
        rows.append(
            {
                "fold": fold,
                "score": _score(est, X.iloc[val_idx], y.iloc[val_idx]),
            }
        )
    df = pd.DataFrame(rows)
    df.attrs["score_mean"] = float(df["score"].mean())
    df.attrs["score_std"] = float(df["score"].std(ddof=1))
    return df
