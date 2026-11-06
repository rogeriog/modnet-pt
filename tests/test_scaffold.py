"""M0 scaffold tests: package imports, sklearn glue, API surface.

The estimator contract is testable before the numerics land: params are
stored verbatim (sklearn convention), get_params/set_params/clone work, and
unimplemented milestones raise NotImplementedError (not silent wrong
answers). M1-M3 paths are now implemented and covered in their own test
modules; this file pins the public contract and defaults.
"""

import numpy as np
import pytest
from sklearn.base import clone

import modnet_pt
from modnet_pt import FRFeatureSelector, MatminerFeaturizer, MODNetRegressor


def test_version():
    assert modnet_pt.__version__ == "0.1.0a0"


def test_public_api_surface():
    for name in modnet_pt.__all__:
        assert hasattr(modnet_pt, name), name


def test_estimator_params_stored_verbatim():
    """sklearn convention: __init__ must not touch its arguments."""
    neurons = ([64], [32], [16], [16])
    est = MODNetRegressor(n_feat=128, num_neurons=neurons, epochs=50, lr=0.01)
    params = est.get_params()
    assert params["n_feat"] == 128
    assert params["epochs"] == 50
    assert params["num_neurons"] == neurons


def test_estimator_clone_roundtrip():
    est = MODNetRegressor(n_feat=32, bootstrap=3, random_state=7)
    cloned = clone(est)
    assert cloned.get_params() == est.get_params()
    assert cloned is not est


def test_estimator_set_params_chainable():
    est = MODNetRegressor().set_params(n_feat=16, loss="mae")
    assert est.n_feat == 16 and est.loss == "mae"


def test_unimplemented_milestones_raise_notimplemented():
    X = np.zeros((10, 4))
    y = np.zeros(10)
    with pytest.raises(NotImplementedError):
        MODNetRegressor(search="preset").fit(X, y)
    with pytest.raises(NotImplementedError):
        MODNetRegressor(bootstrap=4).fit(X, y)
    with pytest.raises(NotImplementedError):
        MODNetRegressor(uncertainty=True).fit(X, y)


def test_featurizer_preset_validation():
    with pytest.raises(ValueError, match="unknown preset"):
        MatminerFeaturizer(preset="not-a-preset").fit(None, None)


def test_selector_defaults_mirror_upstream():
    sel = FRFeatureSelector()
    assert sel.n_features == 64          # vanilla.py:48 n_feat default
    assert sel.n_neighbors == 3          # preprocessing.py:45 kNN MI default
    assert sel.n_samples == 6000         # preprocessing.py:799 subsample size
    assert sel.drop_thr == 0.2           # preprocessing.py:175 entropy drop


def test_regressor_defaults_mirror_upstream():
    est = MODNetRegressor()
    assert est.num_neurons == ([64], [32], [16], [16])   # vanilla.py:48
    assert est.n_feat == 64
    assert est.lr == 1e-3                                 # vanilla.py:266
    assert est.epochs == 200
    assert est.batch_size == 128
    assert est.xscale == "minmax"
    assert est.loss == "mse"
