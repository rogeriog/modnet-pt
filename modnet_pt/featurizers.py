"""Featurization transformer wrapping matminer.

Port of upstream MODNet featurization (v0.4.4 @ 1281977, plus the pandas-2
fixes from the ppdebreuck fork @ fca6bda0):
  modnet/featurizers/featurizers.py:17-338  (MODFeaturizer orchestration)
  modnet/featurizers/utils.py:6             (clean_df)
  modnet/featurizers/presets/debreuck_2020.py:9-268
  modnet/featurizers/presets/matminer_2023.py:8-250
  modnet/featurizers/presets/matminer_all_2023.py:8-414
Upstream feeds featurized dataframes through a MODData container; here the
transformer owns fit-state (all-NaN / duplicate-column drops learned on the
training set only) so Pipelines stay leak-free. See DESIGN.md §2.1 (M1).
"""

from __future__ import annotations

import contextlib

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

PRESETS = (
    "composition_only_matminer_2023",
    "composition_only_matminer_all_2023",
    "composition_only_debreuck_2020",
    "matminer_2023",
    "matminer_all_2023",
    "debreuck_2020",
)

_SITE_ALIASES = {
    "GeneralizedRadialDistributionFunction": "GeneralizedRDF",
    "AGNIFingerprints": "AGNIFingerPrint",
    "BondOrientationalParameter": "BondOrientationParameter",
}

_CRYSTAL_SYSTEM = {
    "cubic": 1,
    "tetragonal": 2,
    "orthorombic": 3,
    "hexagonal": 4,
    "trigonal": 5,
    "monoclinic": 6,
    "triclinic": 7,
}


def clean_df(df: pd.DataFrame, drop_allnan: bool = True) -> pd.DataFrame:
    """Numeric-only + inf→NaN cleanup, mirroring upstream
    ``modnet.featurizers.utils.clean_df`` (utils.py:6)."""
    df = df.select_dtypes(include="number")
    if drop_allnan:
        df = df.dropna(axis=1, how="all")
    return df.replace([np.inf, -np.inf], np.nan)


def _composition_preset(preset: str) -> bool:
    return preset.startswith("composition_only_")


def _build_preset(preset: str) -> dict:
    """Instantiate the matminer featurizer lists for a preset.

    Mirrors the featurizer tuples of the three upstream preset families
    (debreuck_2020.py:105-150, matminer_2023.py:100-152,
    matminer_all_2023.py:170-280). The ``composition_only_*`` variants keep
    only the composition list, exactly like upstream's ``CompositionOnly``
    subclasses (which also disable the oxidation-state featurizers).
    Imports are deferred so that importing ``modnet_pt`` stays light.
    """
    base = preset[len("composition_only_"):] if _composition_preset(preset) else preset

    with contextlib.redirect_stdout(None):
        from matminer.featurizers.composition import (
            AtomicOrbitals,
            AtomicPackingEfficiency,
            BandCenter,
            CationProperty,
            ElectronAffinity,
            ElectronegativityDiff,
            ElementFraction,
            ElementProperty,
            IonProperty,
            Miedema,
            OxidationStates,
            Stoichiometry,
            TMetalFraction,
            ValenceOrbital,
            WenAlloys,
            YangSolidSolution,
        )
        from matminer.featurizers.site import (
            AGNIFingerprints,
            AverageBondAngle,
            AverageBondLength,
            BondOrientationalParameter,
            ChemEnvSiteFingerprint,
            CoordinationNumber,
            CrystalNNFingerprint,
            GaussianSymmFunc,
            GeneralizedRadialDistributionFunction,
            LocalPropertyDifference,
            OPSiteFingerprint,
            VoronoiFingerprint,
        )
        from matminer.featurizers.structure import (
            BondFractions,
            ChemicalOrdering,
            CoulombMatrix,
            DensityFeatures,
            Dimensionality,
            ElectronicRadialDistributionFunction,
            EwaldEnergy,
            GlobalSymmetryFeatures,
            MaximumPackingEfficiency,
            MinimumRelativeDistances,
            RadialDistributionFunction,
            SineCoulombMatrix,
            StructuralComplexity,
            StructuralHeterogeneity,
            XRDPowderPattern,
        )
        from pymatgen.analysis.local_env import VoronoiNN

    if base == "debreuck_2020" or base == "matminer_2023":
        composition = (
            AtomicOrbitals(),
            AtomicPackingEfficiency(),
            BandCenter(),
            ElementFraction(),
            ElementProperty.from_preset("magpie"),
            IonProperty(),
            Miedema(),
            Stoichiometry(),
            TMetalFraction(),
            ValenceOrbital(),
            YangSolidSolution(),
        )
        oxid = (ElectronegativityDiff(), OxidationStates())
        structure = (
            DensityFeatures(),
            GlobalSymmetryFeatures(),
            RadialDistributionFunction(),
            CoulombMatrix(),
            SineCoulombMatrix(),
            EwaldEnergy(),
            BondFractions(),
            StructuralHeterogeneity(),
            MaximumPackingEfficiency(),
            ChemicalOrdering(),
            XRDPowderPattern(),
        )
        site = (
            AGNIFingerprints(),
            AverageBondAngle(VoronoiNN()),
            AverageBondLength(VoronoiNN()),
            BondOrientationalParameter(),
            ChemEnvSiteFingerprint.from_preset("simple"),
            CoordinationNumber(),
            CrystalNNFingerprint.from_preset("ops"),
            GaussianSymmFunc(),
            GeneralizedRadialDistributionFunction.from_preset("gaussian"),
            LocalPropertyDifference(),
            OPSiteFingerprint(),
            VoronoiFingerprint(),
        )

    elif base == "matminer_all_2023":
        # matminer_2023 lists plus the "all" extras (pymatgen/deml element
        # data, WenAlloys, cation/electron-affinity oxidation featurizers,
        # extra structure featurizers), mirroring matminer_all_2023.py:170-280.
        from matminer.utils.data import DemlData, PymatgenData

        pymatgen_features = [
            "atomic_mass",
            "X",
            "row",
            "group",
            "atomic_radius",
            "molar_volume",
            "boiling_point",
            "melting_point",
            "density",
            "electrical_resistivity",
            "velocity_of_sound",
            "thermal_conductivity",
            "bulk_modulus",
            "thermal_expansion",
        ]
        deml_features = [
            "atom_num",
            "atom_mass",
            "row_num",
            "col_num",
            "atom_radius",
            "molar_volume",
            "melting_point",
            "boiling_point",
            "first_ioniz",
            "second_ioniz",
            "electron_affin",
            "electronegativity",
            "GGAU_Etot",
            "mus_fere",
            "FERE correction",
        ]
        composition = (
            AtomicOrbitals(),
            AtomicPackingEfficiency(),
            BandCenter(),
            ElementFraction(),
            ElementProperty.from_preset("magpie"),
            ElementProperty(
                data_source=PymatgenData(),
                stats=["minimum", "maximum", "range", "mean", "std_dev"],
                features=pymatgen_features,
            ),
            ElementProperty(
                data_source=DemlData(),
                stats=["minimum", "maximum", "range", "mean", "std_dev"],
                features=deml_features,
            ),
            Miedema(),
            Stoichiometry(),
            TMetalFraction(),
            ValenceOrbital(props=["frac"]),
            WenAlloys(),
        )
        oxid = (
            CationProperty.from_preset("deml"),
            ElectronAffinity(),
            ElectronegativityDiff(),
            IonProperty(),
            OxidationStates(),
        )
        structure = (
            BondFractions(),
            ChemicalOrdering(),
            DensityFeatures(),
            Dimensionality(),
            ElectronicRadialDistributionFunction(),
            EwaldEnergy(),
            GlobalSymmetryFeatures(),
            MaximumPackingEfficiency(),
            MinimumRelativeDistances(),
            RadialDistributionFunction(),
            SineCoulombMatrix(),
            StructuralComplexity(),
            StructuralHeterogeneity(),
            XRDPowderPattern(),
        )
        site = (
            AGNIFingerprints(),
            AverageBondAngle(VoronoiNN()),
            AverageBondLength(VoronoiNN()),
            BondOrientationalParameter(),
            ChemEnvSiteFingerprint.from_preset("simple"),
            CoordinationNumber(),
            CrystalNNFingerprint.from_preset("ops"),
            GaussianSymmFunc(),
            GeneralizedRadialDistributionFunction.from_preset("gaussian"),
            LocalPropertyDifference(),
            OPSiteFingerprint(),
            VoronoiFingerprint(),
        )
    else:
        raise ValueError(f"unknown preset {preset!r}; choose from {PRESETS}")

    if _composition_preset(preset):
        return {"composition": composition, "oxid": (), "structure": (), "site": ()}
    return {
        "composition": composition,
        "oxid": oxid,
        "structure": structure,
        "site": site,
    }


def _apply_featurizers(work, featurizers, column, n_jobs, ignore_errors):
    """Fit+apply a list of matminer featurizers to a work-frame column.

    Mirrors ``MODFeaturizer._fit_apply_featurizers`` (featurizers.py:103)
    in "multi" mode: fit each featurizer to the column, then apply through a
    ``MultipleFeaturizer`` with multiindex columns flattened to the upstream
    ``Featurizer|feature`` naming. Returns only the feature columns.
    """
    from matminer.featurizers.base import MultipleFeaturizer

    # copy: matminer's featurize_dataframe(multiindex=True) rewrites the
    # input frame's columns in place
    work = work.copy()
    multiple = MultipleFeaturizer([feat.fit(work[column]) for feat in featurizers])
    if n_jobs is not None:
        multiple.set_n_jobs(n_jobs)
    out = multiple.featurize_dataframe(
        work,
        column,
        multiindex=True,
        ignore_errors=ignore_errors,
    )
    out = out.rename(columns={"Input Data": ""})
    out.columns = out.columns.map("|".join).str.strip("|")
    return out.drop(columns=[c for c in work.columns if c in out.columns])


def _featurize_oxid_part(work, oxid_featurizers, n_jobs, ignore_errors):
    """Oxidation-state featurization, mirroring the oxid section of
    ``MODFeaturizer.featurize_composition`` (featurizers.py:196-232):
    integer-computation handling, ``CompositionToOxidComposition``, then the
    oxidation featurizers on "composition_oxid". Returns only features."""
    from matminer.featurizers.conversions import CompositionToOxidComposition
    from pymatgen.core import Composition

    df = work.copy()
    col_comp = "composition"
    if not all(
        all(amt == int(amt) for amt in comp.values())
        for comp in df["composition"].values
    ):
        df = df.copy()
        df["integer_composition"] = [
            Composition(comp.get_integer_formula_and_factor(max_denominator=100)[0])
            for comp in df["composition"].values
        ]
        col_comp = "integer_composition"
    df = CompositionToOxidComposition().featurize_dataframe(
        df, col_id=col_comp, ignore_errors=ignore_errors
    )
    out = _apply_featurizers(
        df, oxid_featurizers, "composition_oxid", n_jobs, ignore_errors
    )
    return out.drop(
        columns=[
            c
            for c in ("composition", "integer_composition", "composition_oxid")
            if c in out.columns
        ]
    )


def _featurize_site_part(work, site_featurizers, n_jobs, ignore_errors):
    """Site featurization via ``SiteStatsFingerprint``, mirroring
    ``featurize_site`` (featurizers.py:293): each site featurizer is wrapped
    with stats ("mean", "std_dev") and columns are prefixed/aliased to the
    upstream names."""
    from matminer.featurizers.structure import SiteStatsFingerprint

    df = work.copy()
    df.columns = ["Input data|" + x for x in df.columns]
    for fingerprint in site_featurizers:
        name = fingerprint.__class__.__name__
        if name == "SOAP":
            fingerprint.fit(df["Input data|structure"])
        stats_fp = SiteStatsFingerprint(fingerprint, stats=("mean", "std_dev"))
        df = stats_fp.featurize_dataframe(
            df, "Input data|structure", multiindex=False, ignore_errors=ignore_errors
        )
        alias = _SITE_ALIASES.get(name, name)
        if "|" not in alias:
            alias += "|"
        df.columns = [
            f"{alias}{x}" if "|" not in x else x for x in df.columns
        ]
    out = df.drop(columns=["Input data|structure"], errors="ignore")
    # drop all-zero site columns (matminer_2023.py:233)
    return out.loc[:, (out != 0).any(axis=0)]


def _postprocess(df, preset):
    """Per-preset column recoding mirroring the upstream preset subclasses
    (debreuck_2020.py:154-247, matminer_2023.py:154-235)."""
    from pymatgen.core.periodic_table import Element

    if "AtomicOrbitals|HOMO_character" in df.columns:
        orbitals = {"s": 1, "p": 2, "d": 3, "f": 4}
        df["AtomicOrbitals|HOMO_character"] = df["AtomicOrbitals|HOMO_character"].map(
            orbitals
        )
        df["AtomicOrbitals|LUMO_character"] = df["AtomicOrbitals|LUMO_character"].map(
            orbitals
        )
        df["AtomicOrbitals|HOMO_element"] = df["AtomicOrbitals|HOMO_element"].apply(
            lambda x: -1 if not isinstance(x, str) else Element(x).Z
        )
        df["AtomicOrbitals|LUMO_element"] = df["AtomicOrbitals|LUMO_element"].apply(
            lambda x: -1 if not isinstance(x, str) else Element(x).Z
        )

    if "GlobalSymmetryFeatures|crystal_system" in df.columns:
        df["GlobalSymmetryFeatures|crystal_system"] = df[
            "GlobalSymmetryFeatures|crystal_system"
        ].map(_CRYSTAL_SYSTEM)
        df["GlobalSymmetryFeatures|is_centrosymmetric"] = (
            df["GlobalSymmetryFeatures|is_centrosymmetric"]
            .astype(float)
            .fillna(0.0)
        )
    return df


class MatminerFeaturizer(TransformerMixin, BaseEstimator):
    """Turn pymatgen compositions/structures into matminer descriptor columns.

    Parameters mirror the upstream preset registry; ``preset`` selects which
    featurizer set runs. ``fit`` featurizes the training data and learns the
    all-NaN and duplicate-column drops; ``transform`` applies the same
    featurization and keeps the learned columns, so Pipelines stay leak-free
    (improvement over upstream ``MODData.featurize``, which cleans whatever
    data it sees).

    Input: DataFrame with a ``"composition"`` (pymatgen ``Composition``) or
    ``"structure"`` (pymatgen ``Structure``) column. Output: DataFrame whose
    columns are the matminer features named ``Featurizer|feature``, matching
    upstream naming.
    """

    def __init__(
        self,
        preset: str = "matminer_2023",
        drop_allnan: bool = True,
        drop_duplicates: bool = True,
        n_jobs: int | None = None,
        ignore_errors: bool = True,
    ):
        self.preset = preset
        self.drop_allnan = drop_allnan
        self.drop_duplicates = drop_duplicates
        self.n_jobs = n_jobs
        self.ignore_errors = ignore_errors

    def _featurize(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.preset not in PRESETS:
            raise ValueError(
                f"unknown preset {self.preset!r}; choose from {PRESETS}"
            )
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                "MatminerFeaturizer expects a pandas DataFrame with a "
                "'composition' or 'structure' column."
            )
        lists = _build_preset(self.preset)

        cols = [c for c in ("composition", "structure") if c in X.columns]
        if not cols:
            raise ValueError(
                "Input DataFrame must have a 'composition' or 'structure' column."
            )
        work = X[cols].copy()
        if (lists["oxid"] or lists["structure"] or lists["site"]) and "structure" not in work.columns:
            raise ValueError(
                f"Preset {self.preset!r} requires structures; pass a 'structure' "
                "column or use a composition_only preset."
            )
        if lists["composition"] and "composition" not in work.columns:
            work["composition"] = work["structure"].apply(lambda s: s.composition)

        parts = []
        if lists["composition"]:
            parts.append(
                _apply_featurizers(
                    work, lists["composition"], "composition", self.n_jobs, self.ignore_errors
                )
            )
        if lists["oxid"]:
            parts.append(
                _featurize_oxid_part(work, lists["oxid"], self.n_jobs, self.ignore_errors)
            )
        if lists["structure"]:
            parts.append(
                _apply_featurizers(
                    work, lists["structure"], "structure", self.n_jobs, self.ignore_errors
                )
            )
        if lists["site"]:
            parts.append(
                _featurize_site_part(work, lists["site"], self.n_jobs, self.ignore_errors)
            )

        out = parts[0]
        for part in parts[1:]:
            out = out.join(part, lsuffix="l", rsuffix="r")
        out = _postprocess(out, self.preset)
        return clean_df(out, drop_allnan=self.drop_allnan)

    def fit(self, X, y=None):
        df = self._featurize(X)

        keep = list(df.columns)
        if self.drop_duplicates:
            keep = list(df.T.drop_duplicates().T.columns)
        self.feature_names_ = keep
        self.n_features_out_ = len(keep)
        self._featurized_fit_ = df.loc[:, keep]
        return self

    def transform(self, X):
        check_is_fitted(self, "feature_names_")
        df = self._featurize(X)
        missing = [c for c in self.feature_names_ if c not in df.columns]
        if missing:
            raise ValueError(
                f"Transform input is missing {len(missing)} features seen at "
                f"fit time, e.g. {missing[:3]}."
            )
        return df.loc[:, self.feature_names_]

    def fit_transform(self, X, y=None, **fit_params):
        """Featurize once: sklearn's default ``fit_transform`` would run the
        (expensive) featurization twice."""
        self.fit(X, y)
        out = self._featurized_fit_
        self._featurized_fit_ = None
        return out

    def _more_tags(self):
        return {"no_validation": True}
