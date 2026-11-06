"""MatminerFeaturizer tests (milestone M1).

Composition presets run in the default suite; structure/site presets are
smoke-tested under the ``slow`` marker (featurizing structures is minutes,
not seconds, on one core).
"""

import numpy as np
import pandas as pd
import pytest
from pymatgen.core import Composition, Lattice, Structure
from sklearn.base import clone

from modnet_pt.featurizers import PRESETS, MatminerFeaturizer, clean_df


def _compositions():
    return pd.DataFrame(
        {
            "composition": [
                Composition("Fe2O3"),
                Composition("NaCl"),
                Composition("Si"),
                Composition("LiCoO2"),
                Composition("GaAs"),
                Composition("PbTe"),
                Composition("SrTiO3"),
                Composition("Ni"),
            ]
        }
    )


def _structures():
    a = Lattice.cubic(4.0)
    return pd.DataFrame(
        {
            "structure": [
                Structure(a, ["Si", "Si"], [[0, 0, 0], [0.5, 0.5, 0.5]]),
                Structure(a, ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]]),
                Structure(Lattice.hexagonal(3.2, 5.2), ["C", "C"],
                          [[1 / 3, 2 / 3, 0], [2 / 3, 1 / 3, 0.5]]),
            ]
        }
    )


@pytest.mark.parametrize("preset", ["composition_only_matminer_2023",
                                    "composition_only_debreuck_2020"])
def test_composition_presets_featurize(preset):
    featurizer = MatminerFeaturizer(preset=preset)
    Xt = featurizer.fit_transform(_compositions())
    assert Xt.shape[0] == 8
    assert Xt.shape[1] > 50
    # upstream feature naming: Featurizer|feature
    assert any("|" in c for c in Xt.columns)
    assert "ElementFraction|Fe" in Xt.columns
    # numeric-only output, upstream naming for orbitals recoded to numbers
    assert Xt["AtomicOrbitals|HOMO_character"].dtype.kind in "if"
    # all-NaN columns dropped, no duplicate columns
    assert not Xt.isna().all().any()
    assert not Xt.columns.duplicated().any()


def test_composition_only_matminer_all_2023():
    Xt = MatminerFeaturizer(
        preset="composition_only_matminer_all_2023"
    ).fit_transform(_compositions())
    assert Xt.shape[1] > 100
    assert "WenAlloys|Yang delta" in Xt.columns


def test_fit_learns_drops_transform_applies_them():
    """The leak-free improvement over upstream: drops are learned on the
    training data only and applied verbatim at transform time."""
    train = _compositions()
    featurizer = MatminerFeaturizer(preset="composition_only_matminer_2023")
    Xt = featurizer.fit_transform(train)
    assert list(featurizer.feature_names_) == list(Xt.columns)

    # transform on a subset keeps the fit-time column set
    test = train.iloc[:3]
    Xt_test = featurizer.transform(test)
    assert list(Xt_test.columns) == list(Xt.columns)

    # an all-NaN feature on a smaller test set is kept (filler NaN), not dropped
    assert Xt_test.shape[1] == Xt.shape[1]


def test_transform_missing_feature_raises():
    featurizer = MatminerFeaturizer(preset="composition_only_matminer_2023")
    featurizer.fit(_compositions())
    with pytest.raises(ValueError, match="missing"):
        featurizer.transform(pd.DataFrame({"composition": [Composition("Xe")]}))


def test_requires_composition_or_structure_column():
    with pytest.raises(ValueError, match="composition"):
        MatminerFeaturizer(
            preset="composition_only_matminer_2023"
        ).fit_transform(pd.DataFrame({"foo": [1]}))


def test_structure_preset_rejects_composition_only_input():
    with pytest.raises(ValueError, match="requires structures"):
        MatminerFeaturizer(preset="matminer_2023").fit(_compositions())


def test_all_presets_validated():
    for preset in PRESETS:
        clone(MatminerFeaturizer(preset=preset))
    with pytest.raises(ValueError, match="unknown preset"):
        MatminerFeaturizer(preset="matminer_1999").fit(_compositions())


def test_clean_df_parity():
    df = pd.DataFrame(
        {"a": [1.0, np.nan], "b": [np.inf, 2.0], "c": ["x", "y"],
         "d": [np.nan, np.nan]}
    )
    out = clean_df(df)
    # numeric columns kept (partial NaN is fine, inf → NaN), object and
    # all-NaN columns dropped — upstream clean_df semantics
    assert list(out.columns) == ["a", "b"]
    assert np.isnan(out["b"].iloc[0])


@pytest.mark.slow
def test_matminer_2023_structure_preset_smoke():
    """Smoke: full structure+site featurization produces upstream-named
    features (GlobalSymmetryFeatures recoded to ints, site stats)."""
    featurizer = MatminerFeaturizer(preset="matminer_2023")
    Xt = featurizer.fit_transform(_structures())
    assert Xt.shape[0] == 3
    assert Xt.shape[1] > 500
    assert "GlobalSymmetryFeatures|crystal_system" in Xt.columns
    assert Xt["GlobalSymmetryFeatures|crystal_system"].dtype.kind in "if"
    assert any(c.startswith("AGNIFingerPrint|") for c in Xt.columns)
    assert any(c.startswith("CrystalNNFingerprint|") for c in Xt.columns)


@pytest.mark.slow
def test_debreuck_2020_preset_smoke():
    Xt = MatminerFeaturizer(preset="debreuck_2020").fit_transform(_structures())
    assert Xt.shape[0] == 3
    assert Xt.shape[1] > 500
