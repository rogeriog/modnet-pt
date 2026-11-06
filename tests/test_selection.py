"""FRR feature selection tests (milestone M2).

Parity bar (DESIGN.md §4.1): the ranking must reproduce the pinned upstream
algorithm. The reference implementation here is a verbatim transcription of
upstream ``get_features_dyn`` (preprocessing.py:476-513, v0.4.4 @ 1281977),
so any divergence of the production port (scores, rr_parameters, dtype
fixes) shows up as a test failure.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from modnet_pt.selection import (
    FRFeatureSelector,
    get_cross_nmi,
    get_rr_c_parameter_default,
    get_rr_p_parameter_default,
    merge_ranked,
    nmi_target,
)


def _make_data(n=150, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, 12)), columns=[f"f{i}" for i in range(12)])
    X["f11"] = X["f0"]  # exact duplicate of f0
    X["f12"] = 3.0  # constant
    y = pd.DataFrame(
        {
            "t0": X["f1"] * 2 + X["f2"] * 2.5 + rng.normal(0, 0.05, n),
            "t1": X["f3"] - X["f4"] + rng.normal(0, 0.05, n),
        }
    )
    return X, y


def test_dynamic_rr_parameters_mirror_paper():
    # preprocessing.py:293 / :307 — p = max(0.1, 4.5 - 0.4 n^0.4), c = min(1e5, 1e-6 n^3)
    assert get_rr_p_parameter_default(1) == pytest.approx(4.1)
    assert get_rr_p_parameter_default(10) == pytest.approx(4.5 - 0.4 * 10**0.4)
    assert get_rr_p_parameter_default(1000) == pytest.approx(0.1)
    assert get_rr_c_parameter_default(10) == pytest.approx(1e-3)
    assert get_rr_c_parameter_default(1000) == pytest.approx(1.0e3)
    assert get_rr_c_parameter_default(100000) == 1e5  # capped  # capped


def test_nmi_target_ranks_informative_features_first():
    X, y = _make_data()
    nmi = nmi_target(X, y[["t0"]], random_state=12)
    top = nmi["t0"].nlargest(2).index.tolist()
    assert set(top) == {"f1", "f2"}
    # exact duplicate and constant never survive target-NMI computation
    assert "f11" not in nmi.index
    assert "f12" not in nmi.index


def test_cross_nmi_properties():
    X, _ = _make_data()
    cnmi = get_cross_nmi(X, random_state=12)
    # symmetric, diag 1.0, bounded
    assert np.allclose(cnmi.values, cnmi.values.T)
    assert (np.diag(cnmi.values) == 1.0).all()
    assert ((cnmi.values <= 1.0 + 1e-12) & (cnmi.values >= -1e-12)).all()
    # low-entropy (constant) features dropped
    assert "f12" not in cnmi.index


def test_frr_ranking_matches_upstream_reference():
    """Bit-for-bit ranking parity with a verbatim transcription of upstream
    ``get_features_dyn`` (preprocessing.py:476)."""
    X, y = _make_data(n=60, seed=3)
    sel = FRFeatureSelector(n_features=6, random_state=12).fit(X, y)

    # --- reference: verbatim upstream get_features_dyn ---
    def reference_get_features_dyn(n_feat, cross_nmi, target_nmi):
        missing = [x for x in cross_nmi.index if x not in target_nmi.index]
        cross_nmi = cross_nmi.drop(missing, axis=0).drop(missing, axis=1)
        missing = [x for x in target_nmi.index if x not in cross_nmi.index]
        target_nmi = target_nmi.drop(missing)
        target_nmi = target_nmi.replace([np.inf, -np.inf, np.nan], 0)

        first_feature = target_nmi.nlargest(1).index[0]
        feature_set = [first_feature]
        get_p = get_rr_p_parameter_default
        get_c = get_rr_c_parameter_default

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
            next_feature = score.min(axis=1).idxmax(axis=0)
            feature_set.append(next_feature)
        return feature_set

    ranked = reference_get_features_dyn(
        -1, sel.cross_nmi_, sel.nmi_target_["t0"]
    )
    assert sel.ranking_by_target_["t0"] == ranked


def test_selector_ranking_and_transform():
    X, y = _make_data()
    sel = FRFeatureSelector(n_features=5, random_state=12)
    Xt = sel.fit_transform(X, y)
    assert sel.ranking_[0] in {"f1", "f2", "f3"}  # most relevant for some target
    assert list(Xt.columns) == sel.ranking_[:5]
    assert Xt.shape == (len(X), 5)
    # transform on new data keeps the same ranked columns
    Xt2 = sel.transform(X.iloc[:10])
    assert list(Xt2.columns) == list(Xt.columns)


def test_selector_multi_target_merge():
    X, y = _make_data()
    sel = FRFeatureSelector(n_features=5, random_state=12).fit(X, y)
    assert set(sel.ranking_by_target_) == {"t0", "t1"}
    merged = merge_ranked(
        [sel.ranking_by_target_["t0"], sel.ranking_by_target_["t1"]]
    )
    assert sel.ranking_ == merged
    assert len(set(sel.ranking_)) == len(sel.ranking_)
    # both targets' top features must appear early in the merged ranking
    assert sel.ranking_by_target_["t0"][0] in sel.ranking_[:4]
    assert sel.ranking_by_target_["t1"][0] in sel.ranking_[:4]


def test_selector_rr_parameters_constant():
    X, y = _make_data(n=60, seed=1)
    rr = {"p": {"function": "constant", "value": 1.0},
          "c": {"function": "constant", "value": 1e-3}}
    sel = FRFeatureSelector(n_features=5, rr_parameters=rr, random_state=12)
    sel.fit(X, y)
    assert len(sel.ranking_) >= 5
    with pytest.raises(ValueError, match="both parameters"):
        FRFeatureSelector(rr_parameters={"p": 1.0}).fit(X, y)


def test_selector_deterministic_with_seed():
    X, y = _make_data(n=50, seed=2)
    r1 = FRFeatureSelector(n_features=5, random_state=12).fit(X, y).ranking_
    r2 = FRFeatureSelector(n_features=5, random_state=12).fit(X, y).ranking_
    assert r1 == r2


def test_selector_clone_and_params():
    sel = FRFeatureSelector(n_features=7, drop_thr=0.3)
    cloned = clone(sel)
    assert cloned.get_params() == sel.get_params()


def test_selector_requires_y():
    X, _ = _make_data(n=20)
    with pytest.raises(ValueError, match="requires"):
        FRFeatureSelector().fit(X, None)


def test_selector_subsamples_large_inputs():
    """n_samples caps the (quadratic) NMI cost, mirroring
    MODData.feature_selection (preprocessing.py:867)."""
    rng = np.random.default_rng(5)
    X = pd.DataFrame(rng.normal(size=(30, 6)), columns=[f"g{i}" for i in range(6)])
    y = pd.DataFrame({"t": X["g1"] + rng.normal(0, 0.1, 30)})
    sel = FRFeatureSelector(n_features=3, n_samples=10, random_state=12)
    sel.fit(X, y)
    assert sel.nmi_target_.shape[0] <= 6
    assert sel.ranking_[0] == "g1"
