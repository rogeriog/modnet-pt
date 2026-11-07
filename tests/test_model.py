"""MODNetRegressor tests (milestone M3): fit/predict contract, joint
learning target groups, loss weighting, scaling semantics, save/load."""

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline

from modnet_pt import FRFeatureSelector, MODNetRegressor


def _make_data(n=100, seed=0, n_features=12):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        rng.normal(size=(n, n_features)),
        columns=[f"f{i}" for i in range(n_features)],
    )
    y = pd.DataFrame(
        {
            "t0": X["f1"] * 2 + X["f2"] + rng.normal(0, 0.05, n),
            "t1": X["f3"] - X["f4"] + rng.normal(0, 0.05, n),
        }
    )
    return X, y


def _small_regressor(**kwargs):
    params = {"n_feat": 12, "num_neurons": ([16], [8], [8], [8]),
             "epochs": 60, "batch_size": 32, "lr": 5e-3, "random_state": 0}
    params.update(kwargs)
    return MODNetRegressor(**params)


def test_fit_predict_shapes_and_names():
    X, y = _make_data()
    est = _small_regressor().fit(X, y)
    preds = est.predict(X)
    assert isinstance(preds, pd.DataFrame)
    assert list(preds.columns) == ["t0", "t1"]
    assert preds.shape == y.shape
    assert np.isfinite(preds.values).all()


def test_training_reduces_loss():
    X, y = _make_data()
    est = _small_regressor(val_fraction=0.2).fit(X, y)
    assert est.history_["loss"][-1] < est.history_["loss"][0]
    assert len(est.history_["val_loss"]) == len(est.history_["loss"])


def test_learns_single_target():
    X, y = _make_data()
    est = _small_regressor().fit(X, y[["t0"]])
    preds = est.predict(X)
    mae = mean_absolute_error(y["t0"], preds["t0"])
    assert mae < 0.5 * y["t0"].std()


def test_joint_learning_groups_share_heads():
    """Upstream target hierarchy: [['t0', 't1']] = one shared 2-output head
    inside one group; explicit doubly-nested form also accepted."""
    X, y = _make_data()
    est = _small_regressor(targets=[["t0", "t1"]]).fit(X, y)
    assert est.targets_ == [[["t0", "t1"]]]  # normalized to upstream form
    preds = est.predict(X)
    assert list(preds.columns) == ["t0", "t1"]

    est2 = _small_regressor(targets=[[["t0"], ["t1"]]]).fit(X, y)
    assert est2.targets_ == [[["t0"], ["t1"]]]
    assert est2.predict(X).shape == y.shape


def test_targets_validation():
    X, y = _make_data(n=30)
    with pytest.raises(ValueError, match="not present in y"):
        _small_regressor(targets=[["nope"]]).fit(X, y)
    with pytest.raises(ValueError, match="nested"):
        _small_regressor(targets=["t0"]).fit(X, y)


def test_loss_weights_change_training():
    X, y = _make_data()
    base = _small_regressor().fit(X, y)
    weighted = _small_regressor(weights={"t0": 10.0, "t1": 1.0}).fit(X, y)
    # weighting t0 heavily must not be identical to equal weights
    assert not np.allclose(base.predict(X).values, weighted.predict(X).values)


def test_loss_mae_supported():
    X, y = _make_data(n=60)
    est = _small_regressor(loss="mae").fit(X, y)
    assert np.isfinite(est.predict(X).values).all()
    with pytest.raises(ValueError, match="loss"):
        _small_regressor(loss="huber").fit(X, y)


def test_scaling_semantics():
    """minmax: scaled inputs in [-0.5, 0.5] with NaN → -1 (vanilla.py:388)."""
    X, y = _make_data()
    est = _small_regressor().fit(X, y)
    scaled = est.scale_impute_.transform(X.values)
    assert scaled.min() >= -1.0 - 1e-9
    assert scaled.max() <= 0.5 + 1e-9

    X_nan = X.copy()
    X_nan.iloc[0, 0] = np.nan
    est2 = _small_regressor().fit(X_nan, y)
    scaled2 = est2.scale_impute_.transform(X_nan.values)
    assert np.isfinite(scaled2).all()


def test_remap_out_of_bounds():
    X, y = _make_data()
    est = _small_regressor().fit(X, y)
    lo, hi = y["t0"].min(), y["t0"].max()
    margin = 0.25 * (hi - lo)
    preds = est.predict(X, remap_out_of_bounds=True)
    # with remapping, predictions stay within the padded training range
    assert preds["t0"].min() >= lo - margin - 1e-9
    assert preds["t0"].max() <= hi + margin + 1e-9


def test_save_load_roundtrip(tmp_path):
    X, y = _make_data()
    est = _small_regressor().fit(X, y)
    path = str(tmp_path / "model.pt")
    est.save(path)
    loaded = MODNetRegressor.load(path)
    assert loaded.get_params() == est.get_params()
    assert loaded.target_names_ == est.target_names_
    assert np.allclose(loaded.predict(X).values, est.predict(X).values)


def test_ndarray_input():
    X, y = _make_data(n=40)
    est = _small_regressor().fit(X.values, y.values)
    preds = est.predict(X.values)
    assert preds.shape == y.values.shape


def test_n_feat_validation():
    X, y = _make_data(n=30, n_features=8)
    with pytest.raises(ValueError, match="more features"):
        MODNetRegressor(n_feat=64).fit(X, y)


def test_clone_in_pipeline():
    X, y = _make_data(n=80)
    pipe = Pipeline(
        [
            ("select", FRFeatureSelector(n_features=8, random_state=12)),
            ("model", _small_regressor(n_feat=8)),
        ]
    )
    cloned = clone(pipe)
    cloned.fit(X, y)
    preds = cloned.predict(X)
    assert preds.shape == y.shape
    mae = mean_absolute_error(y["t0"], preds["t0"])
    assert mae < 0.5 * y["t0"].std()


def test_reproducible_with_seed():
    X, y = _make_data(n=60)
    p1 = _small_regressor().fit(X, y).predict(X).values
    p2 = _small_regressor().fit(X, y).predict(X).values
    assert np.allclose(p1, p2)


def test_batchnorm_only_multi_target():
    """vanilla.py:147: BatchNorm layers appear only in multi-target mode."""
    from modnet_pt.model import TrunkMultiHead

    single = TrunkMultiHead(4, [[["a"]]], num_neurons=([8], [8], [8], [8]))
    multi = TrunkMultiHead(4, [[["a"], ["b"]]], num_neurons=([8], [8], [8], [8]))
    single_bn = sum(1 for m in single.trunk if isinstance(m, torch.nn.BatchNorm1d))
    multi_bn = sum(1 for m in multi.trunk if isinstance(m, torch.nn.BatchNorm1d))
    assert single_bn == 0
    assert multi_bn == 1


def test_classifier_logit_heads():
    # classification heads return raw logits for the loss functions;
    # softmax/sigmoid are applied at predict time (model.py forward)
    from modnet_pt.model import TrunkMultiHead

    net = TrunkMultiHead(
        4, [[["cls"]]], num_classes={"cls": 3},
        num_neurons=([8], [8], [8], [8]),
    ).double()
    out = net(torch.zeros(5, 4).double())
    assert out[0].shape == (5, 3)
    # logits are unconstrained (softmax applied later)
    assert not torch.allclose(out[0], torch.softmax(out[0], dim=-1))
