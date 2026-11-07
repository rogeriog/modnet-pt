"""MODNetClassifier tests (milestone M4): softmax/sigmoid heads,
cross-entropy, predict/predict_proba contracts, early stopping."""

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.metrics import accuracy_score

from modnet_pt import MODNetClassifier


def _classification_data(n=200, seed=0, n_features=6):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        rng.normal(size=(n, n_features)),
        columns=[f"f{i}" for i in range(n_features)],
    )
    label = np.where(
        X["f0"] + X["f1"] > 0.8,
        "high",
        np.where(X["f0"] - X["f2"] < -0.8, "low", "mid"),
    )
    return X, pd.DataFrame({"grade": label})


def _small_classifier(**kwargs):
    params = {
        "n_feat": 6,
        "num_neurons": ([16], [8], [8], [8]),
        "epochs": 80,
        "batch_size": 32,
        "random_state": 0,
    }
    params.update(kwargs)
    return MODNetClassifier(**params)


def test_multiclass_fit_predict():
    X, y = _classification_data()
    clf = _small_classifier().fit(X, y)
    preds = clf.predict(X)
    assert list(preds.columns) == ["grade"]
    assert set(preds["grade"]).issubset({"high", "low", "mid"})
    # must beat the majority-class baseline
    majority = y["grade"].value_counts().max() / len(y)
    assert accuracy_score(y["grade"], preds["grade"]) > majority


def test_predict_proba_contract():
    X, y = _classification_data()
    clf = _small_classifier().fit(X, y)
    proba = clf.predict_proba(X)
    assert proba.shape == (len(X), 3)
    assert list(proba.columns.get_level_values("class")) == [
        "high", "low", "mid"
    ]
    values = proba.values
    assert (values >= 0).all() and (values <= 1).all()
    assert np.allclose(values.sum(axis=1), 1.0)
    # predict agrees with the argmax of predict_proba
    preds = clf.predict(X)
    argmax = proba.columns.get_level_values("class")[
        np.argmax(values, axis=1)
    ]
    assert (preds["grade"].values == argmax.values).all()


def test_integer_labels_and_ndarray_input():
    X, y = _classification_data(n=120, seed=3)
    y_int = (y["grade"] == "high").astype(int)
    clf = _small_classifier().fit(X.values, y_int.values)
    preds = clf.predict(X.values)
    assert set(np.unique(preds.values)) <= {0, 1}
    assert clf.classes_["y0"].tolist() == [0, 1]


def test_multilabel_binary_heads():
    X, _y = _classification_data(n=150, seed=1)
    y2 = pd.DataFrame(
        {
            "is_metal": (X["f3"] > 0).astype(int),
            "is_oxide": (X["f4"] < 0).astype(int),
        }
    )
    clf = _small_classifier(multi_label=True, epochs=60).fit(X, y2)
    preds = clf.predict(X)
    proba = clf.predict_proba(X)
    assert list(preds.columns) == ["is_metal", "is_oxide"]
    assert proba.shape == (len(X), 2)  # one positive-class column per target
    assert ((proba.values >= 0) & (proba.values <= 1)).all()
    assert (preds["is_metal"] == y2["is_metal"]).mean() > 0.9
    assert (preds["is_oxide"] == y2["is_oxide"]).mean() > 0.9


def test_joint_classification_multiple_targets():
    X, y = _classification_data(n=150, seed=2)
    y2 = pd.DataFrame(
        {
            "grade": y["grade"],
            "band": np.where(X["f5"] > 0, "wide", "narrow"),
        }
    )
    clf = _small_classifier(epochs=60).fit(X, y2)
    preds = clf.predict(X)
    assert list(preds.columns) == ["grade", "band"]
    assert set(clf.classes_) == {"grade", "band"}
    assert (preds["band"] == y2["band"]).mean() > 0.85


def test_classifier_validation():
    X, y = _classification_data(n=60, seed=4)
    with pytest.raises(ValueError, match="at least 2 classes"):
        _small_classifier().fit(X, pd.DataFrame({"grade": ["a"] * len(X)}))
    with pytest.raises(ValueError, match="not present in y"):
        _small_classifier(targets=[["nope"]]).fit(X, y)


def test_classifier_clone_and_params():
    clf = MODNetClassifier(n_feat=8, multi_label=True, patience=5)
    cloned = clone(clf)
    assert cloned.get_params() == clf.get_params()
    assert cloned.multi_label and cloned.patience == 5


def test_early_stopping_patience():
    X, y = _classification_data(n=150, seed=5)
    clf = _small_classifier(
        epochs=500, val_fraction=0.3, patience=5
    ).fit(X, y)
    assert len(clf.history_["val_loss"]) < 500
    assert clf.history_["val_loss"]  # non-empty


def test_training_reduces_loss():
    X, y = _classification_data(n=120, seed=6)
    clf = _small_classifier(val_fraction=0.2).fit(X, y)
    assert clf.history_["loss"][-1] < clf.history_["loss"][0]


def test_save_load_roundtrip(tmp_path):
    X, y = _classification_data(n=120, seed=7)
    clf = _small_classifier(epochs=40).fit(X, y)
    path = tmp_path / "clf.pt"
    clf.save(str(path))
    loaded = MODNetClassifier.load(str(path))
    pd.testing.assert_frame_equal(clf.predict(X), loaded.predict(X))
    pd.testing.assert_frame_equal(
        clf.predict_proba(X), loaded.predict_proba(X)
    )
