"""Bootstrap ensembles + MC-dropout uncertainty tests (milestone M7)."""

import numpy as np
import pandas as pd
import pytest

from modnet_pt import MODNetRegressor


def _data(n=100, seed=0, n_features=6):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        rng.normal(size=(n, n_features)),
        columns=[f"f{i}" for i in range(n_features)],
    )
    y = pd.DataFrame({"y": X["f0"] - X["f1"] + rng.normal(0, 0.05, n)})
    return X, y


def _small(**kwargs):
    params = {
        "n_feat": 6,
        "num_neurons": ([16], [8], [4], [4]),
        "epochs": 40,
        "batch_size": 32,
        "random_state": 0,
    }
    params.update(kwargs)
    return MODNetRegressor(**params)


def test_bootstrap_ensemble_predict_mean_and_std():
    X, y = _data()
    est = _small(bootstrap=3).fit(X, y)
    assert len(est.ensemble_) == 3
    pred = est.predict(X)
    std = est.predict_std(X)
    assert list(pred.columns) == ["y"]
    assert list(std.columns) == ["y"]
    assert pred["y"].notna().all()
    assert (std["y"] >= 0).all()
    # ensemble mean should be at least as good as a typical member
    from sklearn.metrics import mean_absolute_error

    maes = [
        mean_absolute_error(y["y"], m.predict(X)["y"]) for m in est.ensemble_
    ]
    mae_ens = mean_absolute_error(y["y"], pred["y"])
    assert mae_ens <= max(maes) + 0.1


def test_bootstrap_reproducible_with_seed():
    X, y = _data(n=60, seed=1)
    a = _small(bootstrap=2, epochs=15).fit(X, y)
    b = _small(bootstrap=2, epochs=15).fit(X, y)
    pd.testing.assert_frame_equal(a.predict(X), b.predict(X))


def test_predict_std_requires_ensemble():
    X, y = _data(n=40)
    est = _small(epochs=10).fit(X, y)
    with pytest.raises(AttributeError, match="ensemble"):
        est.predict_std(X)


def test_mc_dropout_uncertainty():
    X, y = _data(n=80, seed=2)
    est = _small(dropout=0.3, epochs=30).fit(X, y)
    mean, std = est.predict_uncertainty(X, n_passes=8)
    assert list(mean.columns) == ["y"]
    assert list(std.columns) == ["y"]
    assert mean["y"].notna().all()
    # dropout active → nonzero epistemic spread on at least some samples
    assert (std["y"] > 0).any()
    # deterministic given random_state
    mean2, std2 = est.predict_uncertainty(X, n_passes=8)
    pd.testing.assert_frame_equal(mean, mean2)
    pd.testing.assert_frame_equal(std, std2)


def test_uncertainty_flag_enables_dropout():
    X, y = _data(n=60, seed=3)
    est = _small(uncertainty=True, epochs=20).fit(X, y)
    _, std = est.predict_uncertainty(X, n_passes=6)
    assert (std["y"] > 0).any()


def test_ensemble_save_load_roundtrip(tmp_path):
    X, y = _data(n=60, seed=4)
    est = _small(bootstrap=2, epochs=15).fit(X, y)
    path = tmp_path / "ens.pt"
    est.save(str(path))
    loaded = MODNetRegressor.load(str(path))
    assert len(loaded.ensemble_) == 2
    pd.testing.assert_frame_equal(est.predict(X), loaded.predict(X))
    pd.testing.assert_frame_equal(est.predict_std(X), loaded.predict_std(X))
