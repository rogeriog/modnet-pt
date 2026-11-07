"""Validation utilities tests (DESIGN §2.5): learning_curve, nested_cv,
matbench_cv."""

import numpy as np
import pandas as pd
import pytest

from modnet_pt import MODNetRegressor
from modnet_pt.validation import learning_curve, matbench_cv, nested_cv


def _data(n=80, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        rng.normal(size=(n, 5)), columns=[f"f{i}" for i in range(5)]
    )
    y = pd.DataFrame({"y": X["f0"] + rng.normal(0, 0.1, n)})
    return X, y


def _tiny_est(**kwargs):
    params = {
        "n_feat": 5,
        "num_neurons": ([8], [4], [2], [2]),
        "epochs": 15,
        "batch_size": 16,
        "random_state": 0,
    }
    params.update(kwargs)
    return MODNetRegressor(**params)


def _tiny_presets():
    return [
        {"batch_size": 16, "lr": 0.01, "n_feat": 5,
         "num_neurons": [[8], [4], [2], [2]], "epochs": 10,
         "loss": "mae", "act": "elu", "xscale": "minmax"},
        {"batch_size": 16, "lr": 0.005, "n_feat": 5,
         "num_neurons": [[8], [4], [2], [2]], "epochs": 10,
         "loss": "mae", "act": "elu", "xscale": "standard"},
    ]


def test_learning_curve_shape_and_trend():
    X, y = _data()
    lc = learning_curve(
        _tiny_est(), X, y, sizes=[30, 60], cv=3, random_state=0
    )
    assert list(lc.columns) == [
        "n_train_samples", "score_mean", "score_std",
        "score_min", "score_max",
    ]
    assert lc["n_train_samples"].tolist() == [30, 60]
    # more data → lower or equal MAE on this easy problem
    assert lc.loc[1, "score_mean"] <= lc.loc[0, "score_mean"] + 0.2


def test_nested_cv_requires_search():
    X, y = _data(n=50)
    with pytest.raises(ValueError, match="search"):
        nested_cv(_tiny_est(), X, y, outer=2, inner=2)


def test_nested_cv_runs_inner_search():
    X, y = _data(n=60)
    est = MODNetRegressor(
        search="preset", presets=_tiny_presets(), search_folds=2,
        random_state=0,
    )
    df = nested_cv(est, X, y, outer=2, inner=2)
    assert list(df.columns) == ["fold", "score"]
    assert len(df) == 2
    assert df["score"].notna().all()
    assert "score_mean" in df.attrs


def test_matbench_cv_fixed_splits():
    X, y = _data(n=50)
    df1 = matbench_cv(_tiny_est(epochs=10), X, y, n_splits=3)
    df2 = matbench_cv(_tiny_est(epochs=10), X, y, n_splits=3)
    pd.testing.assert_frame_equal(df1, df2)  # deterministic
    assert len(df1) == 3
    assert "score_mean" in df1.attrs
