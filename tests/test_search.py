"""Preset hyperparameter search tests (milestone M5): gen_presets grid,
inner k-fold validation, refit-on-full-data, sklearn integration."""

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error

from modnet_pt import MODNetClassifier, MODNetRegressor
from modnet_pt.search import gen_presets, kfold_splits


def _reg_data(n=120, seed=0, n_features=8):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        rng.normal(size=(n, n_features)),
        columns=[f"f{i}" for i in range(n_features)],
    )
    y = pd.DataFrame({"y": X["f0"] * 2 - X["f1"] + rng.normal(0, 0.1, n)})
    return X, y


def _tiny_presets(n_feat=8, epochs=30):
    return [
        {
            "batch_size": 16, "lr": 0.01, "n_feat": n_feat,
            "num_neurons": [[8], [4], [2], [2]], "epochs": epochs,
            "loss": "mae", "act": "elu", "xscale": "minmax",
        },
        {
            "batch_size": 16, "lr": 0.001, "n_feat": n_feat,
            "num_neurons": [[16], [8], [4], [4]], "epochs": epochs,
            "loss": "mae", "act": "relu", "xscale": "standard",
        },
    ]


def test_gen_presets_structure():
    presets = gen_presets(n_feat=256, n_samples=500)
    # 3 n_feat values in [13, 256] ∩ {64,128,256} → 64,128,256 → 9 archs
    # × 2 batch sizes × 3 lrs × 1 epoch × 1 loss × 1 act × 2 xscale = 108
    assert len(presets) == 108
    p = presets[0]
    assert set(p) == {
        "batch_size", "lr", "n_feat", "num_neurons", "epochs",
        "loss", "act", "xscale",
    }
    assert p["loss"] == "mae" and p["act"] == "elu"
    assert all(nn >= 1 for block in p["num_neurons"] for nn in block)


def test_gen_presets_classification_and_small_nfeat():
    clf_presets = gen_presets(n_feat=32, n_samples=2000, classification=True)
    assert all(p["loss"] == "categorical_crossentropy" for p in clf_presets)
    assert all(p["batch_size"] == 64 for p in clf_presets)  # n >= 1000
    # tiny feature count must still produce a usable grid (upstream returns [])
    small = gen_presets(n_feat=4, n_samples=100)
    assert len(small) > 0
    assert all(p["n_feat"] <= 4 for p in small)
    assert all(nn >= 1 for p in small for b in p["num_neurons"] for nn in b)


def test_kfold_splits_deterministic():
    a = kfold_splits(50, n_splits=5)
    b = kfold_splits(50, n_splits=5)
    assert [tuple(map(tuple, s)) for s in a] == [tuple(map(tuple, s)) for s in b]
    assert len(a) == 5 and all(len(tr) == 40 for tr, _ in a)
    # n_splits <= 1 coerces to 5 (upstream vanilla.py:571)
    assert len(kfold_splits(50, n_splits=1)) == 5


def test_preset_search_regressor():
    X, y = _reg_data()
    est = MODNetRegressor(
        search="preset", presets=_tiny_presets(), search_folds=3,
        random_state=0,
    )
    est.fit(X, y)
    info = est.preset_search_
    assert info["scores"].shape == (2, 3)
    assert info["best_preset_idx"] in (0, 1)
    assert info["best_preset"]["epochs"] == 30

    pred = est.predict(X)
    assert list(pred.columns) == ["y"]
    mae = mean_absolute_error(y["y"], pred["y"])
    assert mae < 0.5  # learned something on an easy synthetic

    # fitted attrs came from the refit candidate, not the template params
    assert est.history_["loss"]
    assert len(est.feature_names_used_) == 8


def test_preset_search_beats_fixed_bad_params():
    """The grid should be able to pick a better config than a deliberately
    poor fixed one on the same data."""
    X, y = _reg_data(n=150, seed=1)
    searched = MODNetRegressor(
        search="preset", presets=_tiny_presets(epochs=40), search_folds=3,
        random_state=0,
    ).fit(X, y)
    poor = MODNetRegressor(
        n_feat=8, num_neurons=([4], [2], [1], [1]), lr=0.5, epochs=15,
        batch_size=64, random_state=0,
    ).fit(X, y)
    Xte, yte = _reg_data(n=300, seed=99)
    mae_s = mean_absolute_error(yte["y"], searched.predict(Xte)["y"])
    mae_p = mean_absolute_error(yte["y"], poor.predict(Xte)["y"])
    assert mae_s <= mae_p + 0.05  # searched is at least competitive


def test_preset_search_classifier():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(100, 6)),
                     columns=[f"f{i}" for i in range(6)])
    y = pd.DataFrame({"cls": np.where(X["f0"] > 0, "a", "b")})
    presets = [
        {"batch_size": 16, "lr": 0.01, "n_feat": 6,
         "num_neurons": [[8], [4], [2], [2]], "epochs": 25,
         "act": "elu", "xscale": "minmax"},
        {"batch_size": 16, "lr": 0.001, "n_feat": 6,
         "num_neurons": [[8], [4], [2], [2]], "epochs": 25,
         "act": "relu", "xscale": "standard"},
    ]
    clf = MODNetClassifier(
        search="preset", presets=presets, search_folds=3, random_state=0,
    ).fit(X, y)
    assert clf.preset_search_["scores"].shape == (2, 3)
    pred = clf.predict(X)
    assert set(pred["cls"]).issubset({"a", "b"})
    proba = clf.predict_proba(X)
    assert np.allclose(proba.values.sum(axis=1), 1.0)


def test_preset_search_clone_safe_and_params_roundtrip():
    X, y = _reg_data(n=60)
    est = MODNetRegressor(search="preset", presets=_tiny_presets(epochs=10),
                          search_folds=2, random_state=0)
    clone(est)  # must not raise
    est.fit(X, y)
    params = est.get_params()
    assert params["search"] == "preset"
    assert params["search_folds"] == 2
    assert params["presets"] is est.presets
